import os
import sys
import warnings
from argparse import ArgumentParser
from itertools import combinations
from typing import Dict, Optional, Tuple, List, Callable, Any, Union

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
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


def do_evaluate(
        args: ArgumentParser, submission: pd.DataFrame, eval_metric_func: Callable, label: str = "Cover") -> float:
    image, kind = args.shared_indices
    df = submission.reset_index()[[kind, image, label]]
    df = df.loc[df[kind].isin(args.labels)]
    if df.empty:
        print(f"Warning: No Ground Truth to evaluate; Return 0")
        return 0.

    return eval_metric_func((df[kind] != label).values, (1. - df[label]).values)


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
        args, stacker, params, eval_metric_func: Callable, data: Dict[str, Union[pd.DataFrame, pd.Series]],
        train_on_validation: bool = True, use_update_model: bool = False):
    if not train_on_validation and not use_update_model:
        stacker = stacker(**params)
        stacker.fit(data["train_x"], data["train_y"])
    elif train_on_validation and not use_update_model:
        stacker = stacker(**params)
        stacker.fit(data["valid_x"], data["valid_y"])
    elif use_update_model:
        base_model = stacker(**params)
        base_model.fit(data["train_x"], data["train_y"])
        hparams = params.copy()
        hparams.update({"refresh_leaf": 1, "updater": "refresh", "process_type": "update",})
        stacker = stacker(**hparams)
        stacker.fit(data["valid_x"], data["valid_y"], xgb_model=base_model.get_booster())

    score = eval_metric_func(data["valid_y"], stacker.predict_proba(data["valid_x"])[:, 1])

    file_path = os.path.join(args.output_dir, f"submission_stacker_metric_{score:.06f}_tr.csv")
    if train_on_validation and not use_update_model:
        file_path = os.path.join(args.output_dir, f"submission_stacker_metric_{score:.06f}_val.csv")
    if use_update_model:
        file_path = os.path.join(args.output_dir, f"submission_stacker_metric_{score:.06f}_update_val.csv")

    subm = pd.DataFrame({"Label": stacker.predict_proba(data["test_x"])[:, 1]}, index=data["test_x"].index.rename("Id"))
    print(f"\nSubmission file: {file_path}\nStats:\n{subm.describe()}\nHead:\n{subm.head()}")
    subm.to_csv(file_path)
    return stacker, subm


# Calib: sklearn.isotonic.IsotonicRegression
# GPSINIFF
# Stacking

model_gen = {
    "CatBoostClassifier": CatBoostClassifier,
    "LGBMClassifier": LGBMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "StackingClassifier": StackingClassifier,
    "XGBClassifier": XGBClassifier,
}


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


def main(args: ArgumentParser):
    seed_everything(args.init_seed)
    args = configure_arguments(args)

    configs = initialize_configs(args.configs).configs

    configs["image_proba"] = check_and_filter_proba_files(args, configs["image_proba"])

    eval_metric_func = alaska_weighted_auc
    train_indices, valid_indices = split_train_valid_data(args=args, splitter=StratifiedKFold(n_splits=5), nr_fold=1)
    if args.proba_single:
        for basename in configs["image_proba"]:
            scoring_single_proba_file(args, basename, eval_metric_func, train_indices, valid_indices)
        return

    ret_files = configs["image_proba"]
    ret = [pd.read_parquet(os.path.join(args.cached_dir, basename)) for basename in ret_files]

    if args.proba_combinations:
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

    if args.generate_proba_file:
        ret = [pd.read_parquet(os.path.join(args.cached_dir, basename)) for basename in ret_files]
        df = pd.concat(ret, axis=0).groupby(level=args.shared_indices).mean()
        stats, score = get_inference_file_score(
            args, data=df, eval_metric_func=eval_metric_func, train_indices=train_indices, valid_indices=valid_indices)
        print(f"\ninference file:\n{stats}")
        file_path = f"proba__{args.proba_filename_stem}__metric_{score:.4f}.parquet"
        file_path = os.path.join(args.cached_dir, file_path)
        print(f"generate new proba file and save to: {file_path}")
        df.to_parquet(file_path)
        return

    scikit_parameters_repos = configs["scikit_parameters_repos"]
    data = generate_stacking_data_split(args, configs, train_indices, valid_indices)
    # HPO
    # configs = scikit_parameters_repos["LinearXGBClassifier"]
    # configs = scikit_parameters_repos["LGBMClassifier"]
    if args.model_stacking:
        scikit_model_params = scikit_parameters_repos[args.model]
        estimator = model_gen.get(scikit_model_params["estimator_gen"])
        params = scikit_model_params.get("params")
        print(f"use model: {args.model}: {estimator}")
        if args.refresh or not params:
            init_params = scikit_model_params["init_params"]
            search_space = scikit_model_params["search_space"]
            solver = OptunaTuner(
                init_params=init_params, search_space=search_space, estimator=estimator,
                eval_metric_func=eval_metric_func, n_startup_trials=100, n_trials=250, )
            solver.search(data)
            params = solver.best_params_

        generate_stacked_submission(
            args, stacker=estimator, params=params, eval_metric_func=eval_metric_func, data=data,
            train_on_validation=False)
        generate_stacked_submission(
            args, stacker=estimator, params=params, eval_metric_func=eval_metric_func, data=data,
            train_on_validation=True)

        if args.use_update_model:
            generate_stacked_submission(
                args, stacker=estimator, params=params, eval_metric_func=eval_metric_func, data=data,
                train_on_validation=False, use_update_model=args.use_update_model)

        return

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
    default_model_name: str = "LGBMClassifier"

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="folder for output")
    parser.add_argument("--cached-dir", type=str, default=default_cached_dir, help="folder for cached data")
    parser.add_argument("--meta-dir", type=str, default=default_meta_dir, help="folder for meta data")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="folder for models")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="folder for data")
    #
    parser.add_argument("--eval-metric", type=str, default=default_eval_metric_name, help="eval metric name")
    # configs
    parser.add_argument("--configs", type=str, default=default_configs, help="configs for stacker")
    #
    parser.add_argument(
        "--proba-filename-stem", type=str, default=None, help="filename for the generated proba file")
    parser.add_argument(
        "--generate-proba-file", action="store_true", default=False, help="generate a new proba file from configs")
    parser.add_argument(
        "--proba-single", action="store_true", default=False, help="generate submission for each single proba file")
    parser.add_argument(
        "--proba-combinations", action="store_true", default=False,
        help="generate submissions for the average proba in every combinations of the proba files")
    parser.add_argument(
        "--use-update-model", action="store_true", default=False,
        help="use model having refit option")
    parser.add_argument(
        "--model-stacking", action="store_true", default=False,
        help="generate submissions for every combinations of the proba files")
    parser.add_argument(
        "--model", type=str, default=default_model_name, help="model for stacking")
    #
    parser.add_argument("--refresh", action="store_true", default=False, help="refresh cached data")
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
