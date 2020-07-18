import os
import sys
import warnings
from argparse import ArgumentParser
from itertools import combinations
from typing import Dict, Optional, Tuple, List, Callable, Any, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

EXTERNAL_UTILS_LIB = "../alaska_utils"
sys.path.append(EXTERNAL_UTILS_LIB)

from alaska_utils import alaska_weighted_auc
from alaska_utils import safe_mkdir
from alaska_utils import seed_everything
from alaska_utils import initialize_configs
from alaska_utils import split_train_valid_data
from alaska_utils import configure_arguments
from alaska_utils import generate_submission


def check_and_filter_proba_files(args: ArgumentParser, files_list: List[str]) -> List[str]:
    ret_files = list()
    for i, basename in enumerate(files_list):
        file_path = os.path.join(args.cached_dir, basename)
        if not os.path.exists(file_path):
            print(f"file_path does not exist: {file_path}")
            continue

        ret_files.append(basename)

    return ret_files


def generate_stacking_data_split(
        args: ArgumentParser, configs,
        train_indices: pd.DataFrame, valid_indices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    image, kind = args.shared_indices

    df_meta_info = pd.read_parquet(os.path.join(args.cached_dir, configs["image_stats"]))

    ret = list()
    for i, s in enumerate(configs["image_proba"]):
        basename = s.split("__")
        arch_name: str = basename[1]
        metrics: str = basename[2]

        file_path: str = os.path.join(args.cached_dir, s)
        df = pd.read_parquet(file_path)
        df.rename(columns={c: "_".join([arch_name, metrics, c]) for c in df.columns}, inplace=True)
        print(f"read {df.shape} from {file_path}")
        ret.append(df)

    df_proba = pd.concat(ret, axis=1)
    df_proba = df_proba.join(df_meta_info)
    columns = df_proba.columns.tolist() + [args.col_image_quality]

    data = dict()
    df_train = train_indices.join(df_proba, how='left', on=args.shared_indices).set_index(args.shared_indices)
    data["train_x"] = df_train.reindex(columns=columns)
    data["train_y"] = (df_train[args.col_enum_class] > .1).astype(np.int32)
    data["train_groups"] = df_train[[args.col_image_quality]]

    df_valid = valid_indices.join(df_proba, how='left', on=args.shared_indices).set_index(args.shared_indices)
    data["valid_x"] = df_valid.reindex(columns=columns)
    data["valid_y"] = (df_valid[args.col_enum_class] > .1).astype(np.int32)
    data["valid_groups"] = df_valid[[args.col_image_quality]]

    df_test = pd.read_parquet(args.file_path_test_images_info)
    df_test = df_test.reset_index().join(df_proba, how='left', on=args.shared_indices)
    data["test_x"] = df_test.set_index(image).reindex(columns=columns)
    print(f"Using Features {len(columns)}: {columns}")
    # assert data["valid_x"].columns == data["test_x"].columns
    return data


def get_inference_file_score(
        args: ArgumentParser, data: pd.DataFrame, train_indices: pd.DataFrame, valid_indices: pd.DataFrame,
        eval_metric_func: Callable, label: str = "Cover") -> Tuple[pd.DataFrame, float]:
    image, kind = args.shared_indices

    df_train = train_indices.join(data, how='left', on=args.shared_indices)
    df_valid = valid_indices.join(data, how='left', on=args.shared_indices)

    queue: List[Tuple[str, pd.DataFrame]] = [("train_split", df_train), ("valid_split", df_valid), ]
    ret_df: List[pd.DataFrame] = list()
    ret_score: List[float] = list()
    for name, df in queue:
        stats = df.groupby(args.col_image_quality).apply(
            lambda x: eval_metric_func((x[kind] != label).values, (1. - x[label]).values))
        ret_df.append(stats.rename(name))
        ret_score.append(eval_metric_func((df[kind] != label).values, (1. - df[label]).values))

    df = pd.concat(ret_df, axis=1).T
    df["all_quality"] = ret_score
    return df, ret_score[-1]


def scoring_single_proba_file(
        args: ArgumentParser, file_path: str, eval_metric_func: Callable, train_indices: List, valid_indices: List):
    arch_name: str = file_path.split("__")[1]
    file_path: str = os.path.join(args.cached_dir, file_path)

    df_proba = pd.read_parquet(file_path)
    stats, score = get_inference_file_score(
        args, data=df_proba, eval_metric_func=eval_metric_func, train_indices=train_indices,
        valid_indices=valid_indices)
    print(f"inference file: {file_path}:\n{stats}")

    df = get_test_sub(args, df_proba)
    df.to_csv(os.path.join(args.output_dir, f"submission_{arch_name}_{score:.06f}.csv"))
    return


def get_test_sub(args: ArgumentParser, df_proba: pd.DataFrame) -> pd.DataFrame:
    image, kind = args.shared_indices
    df_test = df_proba.reset_index()
    df_test = df_test.loc[~df_test[kind].isin(args.labels)]
    return generate_submission(args=args, submission=df_test)


def generate_stacked_submission(
        args, stacker, eval_metric_func: Callable, data: Dict[str, Union[pd.DataFrame, pd.Series]],
        train_on_validation: bool = True):
    if not train_on_validation:
        stacker.fit(data["train_x"], data["train_y"])
    elif train_on_validation:
        stacker.fit(data["valid_x"], data["valid_y"])

    score = eval_metric_func(data["valid_y"], stacker.predict_proba(data["valid_x"])[:, 1])

    file_path = os.path.join(args.output_dir, f"submission_stacker_metric_{score:.06f}_tr.csv")
    if train_on_validation:
        file_path = os.path.join(args.output_dir, f"submission_stacker_metric_{score:.06f}_val.csv")

    subm = pd.DataFrame({"Label": stacker.predict_proba(data["test_x"])[:, 1]}, index=data["test_x"].index.rename("Id"))
    print(f"\nSubmission file: {file_path}\nStats:\n{subm.describe()}\nHead:\n{subm.head()}")
    subm.to_csv(file_path)
    return


# Calib: sklearn.isotonic.IsotonicRegression
# GPSINIFF
# Stacking


import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler


class OptunaTuner:
    def __init__(
            self, eval_metric_func: Callable, estimator, init_params: Optional[Dict],
            search_space: Dict[str, Dict[str, Any]], n_startup_trials: int = 5, n_trials: int = 10,
            greater_is_better: bool = True):
        self.sampler = TPESampler(
            consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=True,
            n_startup_trials=n_startup_trials, n_ei_candidates=24, )
        self.pruner = SuccessiveHalvingPruner(reduction_factor=4, min_early_stopping_rate=0)
        direction = "maximize" if greater_is_better else "minimize"
        self.study = optuna.create_study(
            storage=None, sampler=self.sampler, pruner=self.pruner, study_name="foobar", direction=direction,
            load_if_exists=False)

        self.eval_metric_func: Callable = eval_metric_func
        self.n_trials: int = n_trials
        self.status: bool = False
        #
        self.estimator = estimator
        self.init_params: Dict[str, Any] = init_params
        self.search_space: Dict[str, Dict[str, Any]] = search_space
        #
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.params = self.init_params.copy()

    def _get_suggested_params_from_trail(self, trial) -> Dict:
        suggest_params = dict()
        for k, v in self.search_space.items():
            if v['type'] == 'categorical':
                suggest_params[k] = trial.suggest_categorical(k, v['categorical'])

            elif v['type'] == 'discrete':
                suggest_params[k] = trial.suggest_discrete_uniform(
                    k, low=v['low'], high=v['high'], q=v['step'], )

            elif v['type'] == 'int':
                suggest_params[k] = trial.suggest_int(k, low=v['low'], high=v['high'])

            elif v['type'] == 'loguniform':
                suggest_params[k] = trial.suggest_loguniform(k, low=v['low'], high=v['high'], )

            elif v['type'] == 'uniform':
                suggest_params[k] = trial.suggest_uniform(k, low=v['low'], high=v['high'], )

        return suggest_params

    def search(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.study.optimize(self.objective, n_trials=self.n_trials)
        self.status = True
        trial = self.study.best_trial
        self.params.update(trial.params)
        print(f"best params: {self.params}")
        return self

    def objective(self, trial) -> float:
        params = self.init_params.copy()
        params.update(self._get_suggested_params_from_trail(trial))
        stacker = self.estimator(**params)
        stacker.fit(self.data["train_x"], self.data["train_y"])
        return self.eval_metric_func(self.data["valid_y"], stacker.predict_proba(self.data["valid_x"])[:, 1])

    @property
    def best_params_(self) -> Dict:
        if not self.status:
            raise NotImplementedError()

        print(f"best params: {self.params}")
        return self.params


scikit_parameters_repos = {

    "LinearXGBClassifier": {
        "estimator_gen": XGBClassifier,

        "init_params": {
            "booster": "gblinear",
            "objective": "binary:logistic",
        },

        "search_space": {
            "n_estimators": {"type": "int", "low": 25, "high": 1000},
            "learning_rate": {"type": "loguniform", "low": .001, "high": .1},
            "reg_alpha": {"type": "loguniform", "low": 1e-03, "high": 1.},
            "reg_lambda": {"type": "loguniform", "low": 1e-03, "high": 10.},
            "scale_pos_weight": {"type": "loguniform", "low": 1., "high": 5.},
            "subsample": {"type": "uniform", "low": .5, "high": .95},
        },
    },

    "HistXGBClassifier": {
        "estimator_gen": XGBClassifier,

        "init_params": {
            "base_score": 0.5,
            "booster": "gbtree",
            "colsample_bylevel": 1,
            "colsample_bynode": 1,
            "colsample_bytree": 1,
            "gamma": 0,
            # "importance_type": "gain",
            "learning_rate": 0.1,
            "max_delta_step": 0,
            "max_depth": 5,
            "min_child_weight": 1,
            "missing": None,
            "n_estimators": 500,
            # "n_jobs": N_JOBS,
            "objective": "binary:logistic",
            "random_state": 42,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "seed": 42,
            "silent": True,
            "subsample": 1,
            "verbosity": 0,
            "max_leaves": 7,
            "grow_policy": "lossguide",
            # "tree_method": "gpu_hist",
            "tree_method": "hist",
        },

        "search_space": {
            "colsample_bylevel": {"type": "uniform", "low": .25, "high": .95},
            # "colsample_bynode": {"type": "uniform", "low": .25, "high": .95},
            "colsample_bytree": {"type": "uniform", "low": .10, "high": .50},
            "gamma": {"type": "loguniform", "low": 0.001, "high": 10.},
            "learning_rate": {"type": "loguniform", "low": 0.001, "high": .1},
            "max_delta_step": {"type": "int", "low": 1, "high": 5},
            "max_depth": {"type": "int", "low": 4, "high": 8},
            "min_child_weight": {"type": "int", "low": 1, "high": 24},
            # "n_estimators": {"type": "int", "low": 200, "high": 5000},
            "max_bin": {"type": "categorical", "categorical": [32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 228, 256]},
            "reg_alpha": {"type": "loguniform", "low": 0.001, "high": 1.},
            "reg_lambda": {"type": "loguniform", "low": 0.1, "high": 10.},
            # "scale_pos_weight": {"type": "loguniform", "low": 1., "high": 5.},
            "subsample": {"type": "uniform", "low": .25, "high": .95},
            # "grow_policy": {"type": "categorical", "categorical": ["depthwise", "lossguide"]},
            "max_leaves": {"type": "int", "low": 7, "high": 23},
        },
    },

    "LGBMClassifier": {
        "estimator_gen": LGBMClassifier,

        "init_params": {
            "metric_freq": 100,
            "boosting_type": "gbdt",
            "class_weight": None,
            "colsample_bytree": 1.0,
            "importance_type": "gain",
            "learning_rate": 0.1,
            "max_depth": -1,
            "min_child_samples": 20,
            "min_child_weight": 0.001,
            "min_split_gain": 0.0,
            "n_estimators": 100,
            "n_jobs": -1,
            "num_leaves": 31,
            "objective": None,
            "random_state": None,
            "reg_alpha": 0.0,  # default=0.0
            "reg_lambda": 0.0,  # default=0.0
            "silent": True,
            "subsample": 0.7,  # default=1.0
            "subsample_for_bin": 200000,
            "subsample_freq": 1,  # default=0
        },

        "search_space": {
            "colsample_bytree": {"type": "uniform", "low": .2, "high": .9},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
            "max_depth": {"type": "int", "low": 12, "high": 24},
            "min_child_samples": {"type": "int", "low": 10, "high": 250},
            "min_child_weight": {"type": "loguniform", "low": 0.001, "high": 0.1},
            "min_split_gain": {"type": "loguniform", "low": 0.001, "high": 0.1},
            "n_estimators": {"type": "int", "low": 100, "high": 1000},
            "num_leaves": {"type": "int", "low": 63, "high": 255},
            "reg_alpha": {"type": "loguniform", "low": 0.001, "high": 10.},
            "reg_lambda": {"type": "loguniform", "low": 0.001, "high": 10.},
            "subsample": {"type": "uniform", "low": .25, "high": .95},
        },

        "params": {
            "metric_freq": 100, "boosting_type": "gbdt", "class_weight": None, "colsample_bytree": 0.3351013222509879,
            "importance_type": "gain", "learning_rate": 0.026106125987858088, "max_depth": 17, "min_child_samples": 196,
            "min_child_weight": 0.003442235465150888, "min_split_gain": 0.006466058806990055, "n_estimators": 479,
            "n_jobs": -1, "num_leaves": 63, "objective": None, "random_state": None, "reg_alpha": 1.4689709120163557,
            "reg_lambda": 1.4882227344657941, "silent": True, "subsample": 0.736022596429584,
            "subsample_for_bin": 200000, "subsample_freq": 1},

    },

    "DartLGBMClassifier": {
        "estimator_gen": LGBMClassifier,

        "init_params": {
            "metric_freq": 100,
            "boosting_type": "gbdt",
            "class_weight": None,
            "colsample_bytree": 1.0,
            "importance_type": "gain",
            "learning_rate": 0.1,
            "max_depth": -1,
            "min_child_samples": 20,
            "min_child_weight": 0.001,
            "min_split_gain": 0.0,
            "n_estimators": 100,
            "n_jobs": -1,
            "num_leaves": 31,
            "objective": None,
            "random_state": None,
            "reg_alpha": 0.0,  # default=0.0
            "reg_lambda": 0.0,  # default=0.0
            "silent": True,
            "subsample": 0.7,  # default=1.0
            "subsample_for_bin": 200000,
            "subsample_freq": 1,  # default=0
            # dart
            "drop_rate": 0.1,  # default=0.1
            "max_drop": 50,  # default=50
            "skip_drop": .5,  # defalt=0.5
            "uniform_drop": False,
            "drop_seed": 4,
            "verbosity": -1,
        },

        "search_space": {
            "colsample_bytree": {"type": "uniform", "low": .2, "high": .9},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
            "max_depth": {"type": "int", "low": 12, "high": 24},
            "min_child_samples": {"type": "int", "low": 10, "high": 250},
            "min_child_weight": {"type": "loguniform", "low": 0.001, "high": 0.1},
            "min_split_gain": {"type": "loguniform", "low": 0.001, "high": 0.1},
            "n_estimators": {"type": "int", "low": 100, "high": 1000},
            "num_leaves": {"type": "int", "low": 63, "high": 255},
            "reg_alpha": {"type": "loguniform", "low": 0.001, "high": 10.},
            "reg_lambda": {"type": "loguniform", "low": 0.001, "high": 10.},
            "subsample": {"type": "uniform", "low": .25, "high": .95},
            # dart
            "drop_rate": {"type": "uniform", "low": .05, "high": .15},  # default=0.1
            "max_drop": {"type": "int", "low": 25, "high": 100},  # default = 50,
            "skip_drop": {"type": "uniform", "low": .25, "high": .75},  # default = 0.5,
            "uniform_drop": {"type": "categorical", "categorical": [False, True]},
        },

        "params": {
            "metric_freq": 100, "boosting_type": "gbdt", "class_weight": None, "colsample_bytree": 0.41751254205961874,
            "importance_type": "gain", "learning_rate": 0.010888580762845636, "max_depth": 13, "min_child_samples": 212,
            "min_child_weight": 0.0041204309054919735, "min_split_gain": 0.001872930917084381, "n_estimators": 933,
            "n_jobs": -1, "num_leaves": 66, "objective": None, "random_state": None, "reg_alpha": 3.3540823356593945,
            "reg_lambda": 0.0524168175343131, "silent": True, "subsample": 0.6159663776598608,
            "subsample_for_bin": 200000, "subsample_freq": 1, "drop_rate": 0.05934164873800164, "max_drop": 87,
            "skip_drop": 0.27039555991592634, "uniform_drop": True, "drop_seed": 4, "verbosity": -1}
    },

    "BayesianCatBClassifierGPU": {

        "estimator_gen": CatBoostClassifier,

        "init_params_for_search": {
            "learning_rate": 0.1,
            "loss_function": "MultiClassOneVsAll",
            "eval_metric": "Logloss",  # ["MultiClassOneVsAll", "MultiClass", "CrossEntropy", "Logloss" ],
            # "metric_period": 100,
            "bagging_temperature": 1.0,  # "Bayesian"
            # "subsample": .75,  # "Bernoulli"
            "max_depth": 16,
            "n_estimators": 1000,
            # "colsample_bylevel": None,
            "random_state": 42,
            "reg_lambda": 3.0,
            # "objective": None,
            "max_bin": None,
            "bootstrap_type": "Bayesian",  # ["Bayesian", "Bernoulli", "Poisson", ] # Poisson for GPU,
            "grow_policy": "Lossguide",  # ["SymmetricTree", "Depthwise", "Lossguide"]
            "min_child_samples": 1,
            "max_leaves": None,  # < 63, only for "Lossguide"
            # "one_hot_max_size": None,  #
            "boosting_type": "Plain",  # ["Ordered", "Plain"]
            "logging_level": "Silent",
            "best_model_min_trees": 250,
            # "leaf_estimation_method": "Gradient",  # ["Gradient", "Newton"]
            "task_type": "GPU",
            "devices": "0:1:2",
        },

        "search_space": {
            "bagging_temperature": {"type": "loguniform", "low": 0.1, "high": 100.},  # "Bayesian"
            # "grow_policy": {"type": "categorical", "categorical": ["SymmetricTree", "Depthwise", "Lossguide"]},
            "learning_rate": {"type": "loguniform", "low": 0.001, "high": 0.25},
            "max_depth": {"type": "int", "low": 4, "high": 16},
            "min_child_samples": {"type": "int", "low": 1, "high": 250},
            # "n_estimators": {"type": "int", "low": 200, "high": 5000},
            "max_leaves": {"type": "int", "low": 7, "high": 63},  # only with "Lossguide"
            "reg_lambda": {"type": "loguniform", "low": 0.1, "high": 100.},
            # "subsample": {"type": "uniform", "low": .25, "high": .95},  # "Bernoulli"
            "max_bin": {"type": "categorical", "categorical": [32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 228, 255]},
        },
    },
}



def main(args):
    seed_everything(args.init_seed)
    args = configure_arguments(args)

    configs = initialize_configs(args.configs).configs

    ret_files = check_and_filter_proba_files(args, configs["image_proba"])
    configs["image_proba"] = ret_files

    eval_metric_func = alaska_weighted_auc
    train_indices, valid_indices = split_train_valid_data(args=args, splitter=StratifiedKFold(n_splits=5), nr_fold=1)
    if False:
        for basename in configs["image_proba"]:
            scoring_single_proba_file(args, basename, eval_metric_func, train_indices, valid_indices)

    #
    if args.combinations:
        ret = [pd.read_parquet(os.path.join(args.cached_dir, basename)) for basename in ret_files]
        for i in range(2, len(ret)):
            for j, s in zip(combinations(ret, i), combinations(ret_files, i)):
                df = pd.concat(j, axis=0).groupby(level=args.shared_indices).mean()
                stats, score = get_inference_file_score(
                    args, data=df, eval_metric_func=eval_metric_func, train_indices=train_indices,
                    valid_indices=valid_indices)
                print(f"\ninference file {len(s)}: {s}:\n{stats}")
                df = get_test_sub(args, df)
                df.to_csv(os.path.join(args.output_dir, f"submission_{score:.06f}_ens{len(s)}.csv"))

        return

    data = generate_stacking_data_split(args, configs, train_indices, valid_indices)

    # HPO
    #configs = scikit_parameters_repos["LinearXGBClassifier"]
    #configs = scikit_parameters_repos["LGBMClassifier"]
    configs = scikit_parameters_repos["DartLGBMClassifier"]

    estimator = configs["estimator_gen"]
    init_params = configs["init_params"]
    search_space = configs["search_space"]

    solver = OptunaTuner(
        init_params=init_params, search_space=search_space, estimator=estimator, eval_metric_func=eval_metric_func,
        n_startup_trials=100, n_trials=1000, )
    solver.search(data)
    generate_stacked_submission(
        args, stacker=estimator(**solver.best_params_), eval_metric_func=eval_metric_func, data=data,
        train_on_validation=False)
    generate_stacked_submission(
        args, stacker=estimator(**solver.best_params_), eval_metric_func=eval_metric_func, data=data,
        train_on_validation=True)

    return
    #import pdb;
    #pdb.set_trace()

    #    params = {"booster": "gblinear", "n_estimators": 100, "objective": "binary:logistic", "learning_rate": .05, "colsample_bylevel": .9, "reg_lambda": .005, "reg_alpha": .001}

    #    params = {"booster": "gblinear", "n_estimators": 100, "objective": "binary:logistic", "reg_lambda": 1e-6, "reg_alpha":1e-6, "subsample": .7}

    # stacker.fit(data["valid_x"], data["valid_y"])

    #    params = {"booster": "gbtree", "n_estimators": 1000, "objective": "binary:logistic", "reg_lambda": 1e-6,  "reg_alpha": 1e-6}

    #    params = {"booster": "gblinear", "n_estimators": 100, "objective": "binary:logistic", "colsample_bylevel": .5, "learning_rate": 0.1, "subsample": .7}

    #    stacker = XGBClassifier(**params)
    #    stacker.fit(data["train_x"], data["train_y"])
    #    eval_metric_func(data["valid_y"], stacker.predict_proba(data["valid_x"])[:, 1])

    estimators = list()

    params = {
        "boosting_type": "gbdt", "colsample_bytree": 0.7, "learning_rate": 0.05, "max_depth": 20,
        "min_child_samples": 100, "min_child_weight": 0.002, "min_split_gain": 0.006, "n_estimators": 100,
        "n_jobs": 1, "num_leaves": 124, "random_state": None, "reg_alpha": 0.002, "reg_lambda": 0.002,
        "silent": True, "subsample": 0.65, "subsample_freq": 1
    }
    model = LGBMClassifier(**params)
    estimators.append(("lgbm", model))

    params = {
        "boosting_type": "dart", "colsample_bytree": 0.8, "learning_rate": 0.1, "max_depth": 20,
        "min_child_samples": 10, "min_child_weight": 0.005, "min_split_gain": 0.0025, "n_estimators": 100,
        "n_jobs": 3, "num_leaves": 100, "random_state": None, "reg_alpha": 2.25,
        "reg_lambda": 0.005, "silent": True, "subsample": 0.75, "subsample_freq": 1, "drop_rate": 0.05,
        "max_drop": 35, "skip_drop": 0.7, "uniform_drop": False, "drop_seed": 4, "importance_type": "gain",
    }
    model = LGBMClassifier(**params)
    estimators.append(("dart", model))

    params = {
        "n_estimators": 1000, "max_depth": 8, "max_leaf_nodes": 31, "n_jobs": 3,
        # "class_weight": ["balanced", None, "balanced_subsample"],
    }
    model = RandomForestClassifier(**params)
    estimators.append(("rf", model))

    params = {
        "n_estimators": 1000, "max_depth": 8, "max_leaf_nodes": 31, "n_jobs": 3,
        # "class_weight": ["balanced", None, "balanced_subsample"],
    }
    model = ExtraTreesClassifier(**params)
    estimators.append(("xt", model))

    params = {
        "colsample_bylevel": 0.7, "colsample_bytree": 0.25, "gamma": 0.005, "learning_rate": .05, "max_depth": 6,
        "min_child_weight": 25, "n_estimators": 500, "n_jobs": 3, "subsample": 0.7, "verbosity": 1,
        "silence": True, "tree_method": "hist", "max_bin": 32,
    }
    model = XGBClassifier(**params)
    estimators.append(("xgb", model))

    skf = GroupKFold(n_splits=5)
    for name, m in estimators:
        preds = cross_val_predict(
            m, data["train_x"], y=data["train_y"], groups=data["train_groups"], cv=skf, method='predict_proba',
            n_jobs=args.n_jobs, verbose=1)
        score = alaska_weighted_auc(data["train_y"], preds[:, 1])

        m.fit(data["train_x"], y=data["train_y"])
        print(f"{name}: {score:.6f}\n")
        preds = pd.DataFrame({"Label": m.predict_proba(data["test_x"])[:, 1]}, index=data["test_x"].index, )
        preds.index.name = "Id"
        preds.to_csv(f"submission_{name}_{score:.06f}.csv", index=True)
        print(f"\nSubmission Stats:\n{preds.describe()}\nSubmission:\n{preds.head()}")

    print("procesing stacking")
    stacker = XGBClassifier(**{"booster": "gblinear", "lambda": .005, "alpha": .001})
    #
    clf = StackingClassifier(
        estimators, final_estimator=stacker, cv=skf, stack_method='predict_proba', n_jobs=args.n_jobs,
        passthrough=False, verbose=1).fit(data["train_x"], y=data["train_y"])

    preds = pd.DataFrame({"Label": clf.predict_proba(data["test_x"])[:, 1]}, index=data["test_x"].index, )
    preds.index.name = "Id"
    preds.to_csv("submission_stacked.csv", index=True)
    print(f"\nSubmission Stats:\n{preds.describe()}\nSubmission:\n{preds.head()}")
    return


if "__main__" == __name__:
    #
    default_output_dir: str = "../input/alaska2-image-steganalysis-output/"
    default_cached_dir: str = "../input/alaska2-image-steganalysis-cached-data/"
    default_meta_dir: str = "../input/alaska2-image-steganalysis-image-quality/"
    default_model_dir: str = "../input/alaska2-image-steganalysis-models/"
    default_data_dir: str = "../input/alaska2-image-steganalysis/"
    #
    default_configs: str = "../configs/stacking_baseline.py"
    #
    default_n_jobs: int = 8
    default_init_seed: int = 42
    #
    default_eval_metric_name: str = "weighted_auc"

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="folder for output")
    parser.add_argument("--cached-dir", type=str, default=default_cached_dir, help="folder for cached data")
    parser.add_argument("--meta-dir", type=str, default=default_meta_dir, help="folder for meta data")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="folder for models")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="folder for data")
    #
    parser.add_argument("--eval-metric", type=str, default=default_eval_metric_name, help="eval metric name")
    # configs
    parser.add_argument("--configs", type=str, default=default_configs, help="configs for stacking")
    #
    parser.add_argument("--combinations", action="store_true", default=False, help="combinations")
    #
    parser.add_argument("--refresh-cache", action="store_true", default=False, help="refresh cached data")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs, help="num worker")
    parser.add_argument("--init-seed", type=int, default=default_init_seed, help="initialize random seed")
    # debug
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    args = parser.parse_args()

    # house keeping
    safe_mkdir(args.output_dir)
    safe_mkdir(args.cached_dir)
    safe_mkdir(args.model_dir)
    # start program
    main(args)
