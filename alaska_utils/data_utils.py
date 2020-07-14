import os
from argparse import ArgumentParser
from glob import glob
from typing import List

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

    return args


def index_train_test_images(args: ArgumentParser):
    image, kind = args.shared_indices

    file_path_image_quality: str = "image_quality.csv"

    file_paths: List[str] = [
        args.file_path_all_images_info, args.file_path_train_images_info, args.file_path_test_images_info]

    if all([os.path.exists(path) for path in file_paths]):
        if not args.refresh_cache:
            return

    file_path_image_quality = os.path.join(args.meta_dir, file_path_image_quality)
    df_quality: pd.DataFrame = pd.DataFrame()
    if os.path.exists(file_path_image_quality):
        df_quality = pd.read_csv(file_path_image_quality).set_index(args.shared_indices)
        print(f"read in image quality file: {df_quality.shape}")
    else:
        print(f"image quality not exist: {file_path_image_quality}")

    # process
    list_all_images: List[str] = list(glob(os.path.join(args.data_dir, "*", "*.jpg")))
    df_train = pd.DataFrame({"file_path": list_all_images})
    df_train[image] = df_train["file_path"].apply(lambda x: os.path.basename(x))
    df_train[kind] = df_train["file_path"].apply(lambda x: os.path.split(os.path.dirname(x))[-1])
    df_train[args.col_enum_class] = df_train[kind].map({kind: label for label, kind in enumerate(args.labels)})
    df_train.drop(columns=["file_path"], inplace=True)
    df_train.set_index(args.shared_indices, inplace=True)

    if not df_quality.empty:
        df_train = df_train.join(df_quality[args.col_image_quality])

    print(f"Columns: {df_train.columns.tolist()}, N Uniques:\n{df_train.nunique()}")
    df_train.sort_values(image, inplace=True)
    df_train.to_parquet(args.file_path_all_images_info)
    df_train.loc[df_train[args.col_enum_class].notnull()].to_parquet(args.file_path_train_images_info)
    df_train.loc[df_train[args.col_enum_class].isnull()].to_parquet(args.file_path_test_images_info)
    return


def split_train_valid_data(args: ArgumentParser, splitter: BaseCrossValidator, nr_fold: int = 1):
    image, kind = args.shared_indices
    label = args.col_enum_class

    df_train = pd.read_parquet(args.file_path_train_images_info).reset_index()
    if args.debug:
        df_train = df_train.iloc[:2000]

    df_train[label] = df_train[label].astype(np.int32)
    df = df_train.loc[(~df_train[image].duplicated(keep="first"))]
    for fold, (train_ind, valid_ind) in enumerate(
            splitter.split(X=df[label], y=df[args.col_image_quality], groups=df[image]), 1):
        if nr_fold == fold:
            print(f"using fold {fold:02d} for train valid data split")
            break

    train_df = df_train.loc[df_train[image].isin(df[image].iloc[train_ind])]
    valid_df = df_train.loc[df_train[image].isin(df[image].iloc[valid_ind])]
    return train_df, valid_df
