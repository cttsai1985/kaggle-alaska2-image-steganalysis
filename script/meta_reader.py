import os
import sys
import warnings
from argparse import ArgumentParser
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import List, Callable, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")

EXTERNAL_UTILS_LIB = "../alaska_utils"
sys.path.append(EXTERNAL_UTILS_LIB)

from alaska_utils import safe_mkdir
from alaska_utils import configure_arguments
from alaska_utils import parse_image_to_dir_basename


def process_image(image_file_path: str, functions: Tuple[str, Callable], ) -> Union[pd.Series, pd.DataFrame]:
    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    tmp = {name: [func(image[:, :, i].flatten()) for i in range(image.shape[-1])] for name, func in functions}
    for name, func in functions:
        tmp[name].append(func(image.flatten()))
    tmp = pd.DataFrame(tmp).stack().swaplevel()
    return tmp


def main(args: ArgumentParser):
    for dir_path in [args.output_dir, args.cached_dir, args.model_dir]:
        safe_mkdir(dir_path)

    args = configure_arguments(args)
    file_path = args.file_path_images_stats
    if os.path.exists(file_path) and not args.refresh:
        print(f"{file_path} exists, skip generates meta info")
        return False

    # process
    list_of_functions: List[Tuple[str, Callable]] = [
        ("mean", np.mean), ("std", np.std), ("kurt", kurtosis), ("skew", skew), ]
    list_all_images: List[str] = list(glob(os.path.join(args.data_dir, "*", "*.jpg")))
    with Pool(processes=args.n_jobs) as p:
        func = partial(process_image, functions=list_of_functions)
        df = pd.concat(list(p.map(func, list_all_images)), axis=1).T.astype(np.float32)
        df.columns = [f"{i}_{m}" for m, i in df.columns]

    df_train = parse_image_to_dir_basename(args, list_all_images, column="file_path")

    # compose return dataframe
    image, kind = args.shared_indices
    df[image] = df_train[image].tolist()
    df[kind] = df_train[kind].tolist()
    df.sort_values(image, inplace=True)
    df.set_index(args.shared_indices, inplace=True)

    df.to_parquet(file_path)
    print(f"Save stats to: {file_path}\n{df.describe().T}")
    return


if "__main__" == __name__:
    #
    default_output_dir: str = "../input/alaska2-image-steganalysis-output/"
    default_cached_dir: str = "../input/alaska2-image-steganalysis-cached-data/"
    default_meta_dir: str = "../input/alaska2-image-steganalysis-image-quality/"
    default_model_dir: str = "../input/alaska2-image-steganalysis-models/"
    default_data_dir: str = "../input/alaska2-image-steganalysis/"
    #
    default_n_jobs: int = 1
    default_init_seed: int = 42

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="folder for output")
    parser.add_argument("--cached-dir", type=str, default=default_cached_dir, help="folder for cached data")
    parser.add_argument("--meta-dir", type=str, default=default_meta_dir, help="folder for meta data")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="folder for models")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="folder for data")
    #
    parser.add_argument("--refresh", action="store_true", default=False, help="refresh cached data")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs, help="num worker")
    # debug
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    args = parser.parse_args()

    # start program
    main(args)
