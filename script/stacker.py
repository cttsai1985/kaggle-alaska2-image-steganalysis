import os
import random
import time
import warnings
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from typing import List, Optional, Callable, Tuple, Any, Dict, Sequence, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm  # pretrained model
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
# Class Balance "on fly" from @CatalystTeam
from catalyst.data.sampler import BalanceClassSampler
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

warnings.filterwarnings("ignore")


def index_train_test_images(args):
    working_dir: str = args.data_dir
    output_dir: str = args.cached_dir
    meta_dir: str = args.meta_dir
    file_path_image_quality: str = "image_quality.csv"

    args.file_path_all_images_info = os.path.join(output_dir, "all_images_info.parquet")
    args.file_path_train_images_info = os.path.join(output_dir, "train_images_info.parquet")
    args.file_path_test_images_info = os.path.join(output_dir, "test_images_info.parquet")
    file_paths: List[str] = [
        args.file_path_all_images_info, args.file_path_train_images_info, args.file_path_test_images_info]

    if all([os.path.exists(path) for path in file_paths]):
        if not args.refresh_cache:
            return

    file_path_image_quality = os.path.join(meta_dir, file_path_image_quality)
    df_quality: pd.DataFrame = pd.DataFrame()
    if os.path.exists(file_path_image_quality):
        df_quality = pd.read_csv(file_path_image_quality).set_index(args.shared_indices)
        print(f"read in image quality file: {df_quality.shape}")
    else:
        print(f"image quality not exist: {file_path_image_quality}")
        raise ValueError()

    # process
    list_all_images: List[str] = list(glob(os.path.join(working_dir, "*", "*.jpg")))
    df_train = pd.DataFrame({"file_path": list_all_images})
    df_train["image"] = df_train["file_path"].apply(lambda x: os.path.basename(x))
    df_train["kind"] = df_train["file_path"].apply(lambda x: os.path.split(os.path.dirname(x))[-1])
    df_train["label"] = df_train["kind"].map({kind: label for label, kind in enumerate(args.labels)})
    df_train.set_index(args.shared_indices, inplace=True)

    if not df_quality.empty:
        df_train = df_train.join(df_quality["quality"])

    print(f"{df_train.nunique()}")
    df_train.sort_values("image", inplace=True)
    df_train.to_parquet(args.file_path_all_images_info)
    df_train.loc[df_train["label"].notnull()].to_parquet(args.file_path_train_images_info)
    df_train.loc[df_train["label"].isnull()].to_parquet(args.file_path_test_images_info)
    return


def initialize_configs(filename: str):
    if not os.path.exists(filename):
        raise ValueError("Spec file {spec_file} does not exist".format(spec_file=filename))

    module_name = filename.split(os.sep)[-1].replace(".", "")

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    args.labels: List[str] = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    args.shared_indices: List[str] = ["image", "kind"]

    seed_everything(args.init_seed)
    index_train_test_images(args)

    if "efficientnet" in args.model_arch:
        model = EfficientNet.from_pretrained(
            args.model_arch, advprop=False, in_channels=3, num_classes=len(args.labels))
    else:
        # "seresnet34", resnext50_32x4d"
        model = timm.create_model(
            args.model_arch, pretrained=True, num_classes=len(args.labels), in_chans=3, drop_rate=.5)

    # loading info for training
    training_configs = None
    validation_configs = None
    training_records = list()
    valid_records = list()
    if not args.inference_only:
        # configs
        validation_configs = BaseConfigs.from_file(file_path=args.valid_configs)
        training_configs = BaseConfigs.from_file(file_path=args.train_configs)
        training_configs.loss = LabelSmoothing(smoothing=.05)

        # split data
        skf = StratifiedKFold(n_splits=5)
        train_df, valid_df = split_data(args=args, splitter=skf)

        #
        training_records = process_images_to_records(args, df=train_df)
        valid_records = process_images_to_records(args, df=valid_df)

    # use lightning
    if args.use_lightning:
        model = BaseLightningModule(
            model, training_configs=training_configs, training_records=training_records,
            valid_configs=validation_configs, valid_records=valid_records, eval_metric_name=args.eval_metric,
            eval_metric_func=alaska_weighted_auc)

        model = training_lightning(args=args, model=model)
        model.freeze()

    # raw
    if args.gpus is not None:
        model = model.cuda()
        device = torch.device("cuda:0")

    if not args.inference_only and not args.use_lightning:
        train_dataset = DatasetRetriever(records=training_records, transforms=training_configs.transforms)
        train_loader = DataLoader(
            train_dataset,
            sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
            batch_size=training_configs.batch_size,
            pin_memory=False,
            drop_last=True,
            num_workers=training_configs.num_workers,
        )
        validation_dataset = DatasetRetriever(records=valid_records, transforms=training_configs.transforms)
        val_loader = DataLoader(
            validation_dataset,
            batch_size=validation_configs.batch_size,
            num_workers=validation_configs.num_workers,
            shuffle=False,
            sampler=SequentialSampler(validation_dataset),
            pin_memory=False,
        )

        fitter = Fitter(model=model, device=device, config=training_configs)
        if args.load_checkpoint and os.path.exists(args.checkpoint_path):
            fitter.load(args.checkpoint_path, model_weights_only=args.load_weights_only)
        # fitter.load(f"{fitter.base_dir}/best-checkpoint-024epoch.bin")
        fitter.fit(train_loader, val_loader)

    # Test
    submission = do_inference(args, model=model)
    score = do_evaluate(submission)
    if args.inference_proba:
        print(f"Inference TTA: {score:.04f}")
        file_path = os.path.join(args.cached_dir, f"proba__arch_{args.model_arch}__metric_{score:.4f}.parquet")
        submission.to_parquet(file_path)
    else:
        print(f"Inference: {score:.04f}")
        df = submission.reset_index()[["image", "Cover"]]
        df.columns = ["Id", "Label"]
        df.set_index("Id", inplace=True)
        df["Label"] = 1. - df["Label"]
        df.to_csv("submission.csv", index=True)
        print(f"\nSubmission Stats:\n{df.describe()}\nSubmission:\n{df.head()}")

    return


def inference_proba(args, configs, dataset: Dataset, model: nn.Module) -> pd.DataFrame:
    data_loader = DataLoader(
        dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers, drop_last=False, )

    model.eval()
    outputs = list()
    result = {k: list() for k in args.shared_indices}
    total_num_batch: int = int(len(dataset) / configs.batch_size)
    for step, (image_kinds, image_names, images) in enumerate(data_loader):
        print(
            f"Test Batch Proba: {step:03d} / {total_num_batch:d}, progress: {100. * step / total_num_batch: .02f} %",
            end="\r")

        result["image"].extend(image_names)
        result["kind"].extend(image_kinds)
        outputs.append(nn.functional.softmax(model(images.cuda()), dim=1).data.cpu())

    y_pred = pd.DataFrame(torch.cat(outputs, dim=0).numpy(), columns=args.labels)
    submission = pd.concat([pd.DataFrame(result), y_pred], axis=1).set_index(args.shared_indices).sort_index()

    print(f"\nFinish Test Proba: {submission.shape}, Stats:\n{submission.describe()}")
    return submission


def do_evaluate(submission: pd.DataFrame, label: str = "Cover") -> float:
    df = submission.reset_index()[["kind", "image", label]]
    df = df.loc[df["kind"].isin(args.labels)]
    df["Label"] = 1. - df[label]
    return alaska_weighted_auc(df["kind"].isin(args.labels[1:]), df["Label"].values)


def do_inference(args, model: nn.Module):
    test_configs = BaseConfigs.from_file(file_path=args.test_configs)
    df_test = pd.read_parquet(args.file_path_test_images_info).reset_index()
    if args.inference_proba:
        df_test = pd.read_parquet(args.file_path_all_images_info).reset_index()

    if args.debug:
        df_test = df_test.iloc[:2000]  # sample(n=2000, random_state=42)

    test_records = process_images_to_records(args, df=df_test)
    if args.tta:
        collect = list()
        for i, tta in enumerate(test_configs.tta_transforms, 1):
            print(f"Inference TTA: {i:02d} / {len(test_configs.tta_transforms):02d} rounds")
            dataset = SubmissionRetriever(records=test_records, transforms=tta, )
            df = inference_proba(args, test_configs, dataset=dataset, model=model)
            collect.append(df)

            score = do_evaluate(df)
            print(f"Inference TTA: {i:02d} / {len(test_configs.tta_transforms):02d} rounds: {score:.04f}")

        df = pd.concat(collect, ).groupby(level=args.shared_indices).mean()
        print(f"\nFinish Test Proba: {df.shape}, Stats:\n{df.describe()}")
        return df

    dataset = SubmissionRetriever(records=test_records, transforms=test_configs.transforms, )
    df = inference_proba(args, test_configs, dataset=dataset, model=model)
    return df


def safe_mkdir(directory: str) -> bool:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"make dir: {directory}")
        return True

    print(f"skip making dir: {directory}")
    return False


if "__main__" == __name__:
    #
    default_output_dir: str = "../input/alaska2-image-steganalysis-output/"
    default_cached_dir: str = "../input/alaska2-image-steganalysis-cached-data/"
    default_meta_dir: str = "../input/alaska2-image-steganalysis-image-quality/"
    default_model_dir: str = "../input/alaska2-image-steganalysis-models/"
    default_data_dir: str = "../input/alaska2-image-steganalysis/"
    #
    default_model_arch: str = "efficientnet-b2"
    #
    default_train_configs: str = "../configs/train_baseline.py"
    default_valid_configs: str = "../configs/valid_baseline.py"
    default_test_configs: str = "../configs/test_baseline.py"
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
    parser.add_argument("--model-arch", type=str, default=default_model_arch, help="model arch")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint")
    parser.add_argument("--load-checkpoint", action="store_true", default=False, help="load checkpoint")
    parser.add_argument("--load-weights-only", action="store_true", default=False, help="load weights only")
    # configs
    parser.add_argument("--train-configs", type=str, default=default_train_configs, help="configs for training")
    parser.add_argument("--valid-configs", type=str, default=default_valid_configs, help="configs for validation")
    parser.add_argument("--test-configs", type=str, default=default_test_configs, help="configs for test")
    # functional
    parser.add_argument("--tta", action="store_true", default=False, help="perform test time augmentation")
    parser.add_argument("--inference-proba", action="store_true", default=False, help="only perform inference proba")
    parser.add_argument("--inference-only", action="store_true", default=False, help="only perform inference")
    parser.add_argument("--use-lightning", action="store_true", default=False, help="using lightning trainer")
    parser.add_argument("--refresh-cache", action="store_true", default=False, help="refresh cached data")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs, help="num worker")
    parser.add_argument("--init-seed", type=int, default=default_init_seed, help="initialize random seed")
    #
    parser.add_argument("--gpus", default=None)
    # debug
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    args = parser.parse_args()

    # house keeping
    safe_mkdir(args.output_dir)
    safe_mkdir(args.cached_dir)
    safe_mkdir(args.model_dir)
    # start program
    main(args)
