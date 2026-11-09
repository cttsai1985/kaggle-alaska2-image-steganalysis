import os
from argparse import ArgumentParser
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


def configure_arguments(args: ArgumentParser) -> ArgumentParser:
    args.labels: List[str] = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    args.shared_indices: List[str] = ["image", "kind"]
    args.col_enum_class: str = "label"
    args.col_image_quality: str = "quality"

    args.file_path_all_images_info = os.path.join(args.cached_dir, "all_images_info.parquet")
    args.file_path_train_images_info = os.path.join(args.cached_dir, "train_images_info.parquet")
    args.file_path_test_images_info = os.path.join(args.cached_dir, "test_images_info.parquet")

    args.file_path_images_stats = os.path.join(args.cached_dir, "images_stats_info.parquet")

    file_path_image_quality: str = "image_quality.csv"
    args.file_path_image_quality = os.path.join(args.meta_dir, file_path_image_quality)
    return args


def parse_image_to_dir_basename(
        args: ArgumentParser, list_all_images: List[str], column: str = "file_path") -> pd.DataFrame:
    image, kind = args.shared_indices
    df = pd.DataFrame({column: list_all_images})
    df[image] = df[column].apply(lambda x: os.path.basename(x))
    df[kind] = df[column].apply(lambda x: os.path.split(os.path.dirname(x))[-1])
    df.drop(columns=[column], inplace=True)
    return df


def index_train_test_images(args: ArgumentParser):
    image, kind = args.shared_indices

    file_paths: List[str] = [
        args.file_path_all_images_info, args.file_path_train_images_info, args.file_path_test_images_info]

    if all([os.path.exists(path) for path in file_paths]):
        if not args.refresh_cache:
            return

    df_quality: pd.DataFrame = pd.DataFrame()
    file_path_image_quality = args.file_path_image_quality
    if os.path.exists(file_path_image_quality):
        df_quality = pd.read_csv(file_path_image_quality).set_index(args.shared_indices)
        print(f"read in image quality file: {df_quality.shape}")
    else:
        print(f"image quality not exist: {file_path_image_quality}")

    # process
    list_all_images: List[str] = list(glob(os.path.join(args.data_dir, "*", "*.jpg")))
    df_train = parse_image_to_dir_basename(args, list_all_images, column="file_path")
    df_train[args.col_enum_class] = df_train[kind].map({kind: label for label, kind in enumerate(args.labels)})
    df_train.set_index(args.shared_indices, inplace=True)

    if not df_quality.empty:
        df_train = df_train.join(df_quality[args.col_image_quality])

    print(f"Columns: {df_train.columns.tolist()}, N Uniques:\n{df_train.nunique()}")
    df_train.sort_values(image, inplace=True)
    df_train.to_parquet(args.file_path_all_images_info)
    df_train.loc[df_train[args.col_enum_class].notnull()].to_parquet(args.file_path_train_images_info)
    df_train.loc[df_train[args.col_enum_class].isnull()].to_parquet(args.file_path_test_images_info)
    return


def split_train_test_data(
        args: ArgumentParser, data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image, kind = args.shared_indices

    if data is None:
        data = pd.read_parquet(args.file_path_all_images_info)

    df = data.reset_index()
    mask = df[kind].isin(args.labels)
    return df.loc[mask], df.loc[~mask]


def split_train_valid_data(
        args: ArgumentParser, splitter: BaseCrossValidator, data: Optional[pd.DataFrame] = None,
        nr_fold: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split Data into Train and Valid"""
    label = args.col_enum_class
    image, kind = args.shared_indices
    image_quality = args.col_image_quality

    if data is None:
        data = pd.read_parquet(args.file_path_train_images_info)

    data = data.reset_index()

    if args.debug:
        data = data.iloc[:2000]

    data[label] = data[label].astype(np.int32)
    df = data.loc[(~data[image].duplicated(keep="first"))]
    for fold, (train_ind, valid_ind) in enumerate(
            splitter.split(X=df[label], y=df[image_quality], groups=df[image]), 1):
        if nr_fold == fold:
            print(f"using fold {fold:02d} for train valid data split", end="\r")
            break

    train_df = data.loc[data[image].isin(df[image].iloc[train_ind])]
    valid_df = data.loc[data[image].isin(df[image].iloc[valid_ind])]
    print(f"using fold {fold:02d} for train valid data split: {train_df.shape}, {valid_df.shape}")
    return train_df, valid_df


def generate_submission(args: ArgumentParser, submission: pd.DataFrame) -> pd.DataFrame:
    """Take Test Predictions for 4 classes to Generate Submission File"""
    image, kind = args.shared_indices
    df = submission.reset_index()[[image, args.labels[0]]]
    df.columns = ["Id", "Label"]
    df.set_index("Id", inplace=True)
    df["Label"] = 1. - df["Label"]
    print(f"\nSubmission Stats:\n{df.describe()}\nSubmission Head:\n{df.head()}")
    return df
