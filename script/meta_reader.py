import os
import warnings
from argparse import ArgumentParser
from functools import partial
from glob import glob
from typing import List, Callable, Tuple, Dict
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
# Class Balance "on fly" from @CatalystTeam
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")


def save_hdf_file(file_path: str, data: Dict[str, pd.DataFrame]):
    with pd.HDFStore(file_path, mode="w") as store:
        print(f"save data ({len(data.keys())}) to: {file_path}")
        for k, v in data.items():
            if v is None or not (isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)):
                print(f"skip save key: {k}")
                continue

            store.put(key=k, value=v)
            print(f"save stats: {k}, shape={v.shape}")

    return True


def load_hdf_file(file_path: str) -> Dict[str, pd.DataFrame]:
    data = dict()
    with pd.HDFStore(file_path, mode="r") as store:
        print(f"load data ({len(store.keys())}) from: {file_path}")
        for k in store.keys():
            df = store.get(k)
            data[k.lstrip('/')] = df
            print(f"load key: {k}, shape={df.shape}")

    return data


def process_image(image_file_path: str, functions: Tuple[str, Callable], ) -> Dict[str, pd.Series]:
    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    tmp = pd.DataFrame(
        {name: [func(image[:, :, i].flatten()) for i in range(image.shape[-1])] for name, func in
         functions}).stack().swaplevel()
    return tmp


def main(args):
    args.labels: List[str] = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    shared_indices: List[str] = ["image", "kind"]
    list_of_functions: Tuple[str, Callable] = [
        ("mean", np.mean), ("std", np.std), ("kurt", kurtosis), ("skew", skew), ]

    file_path = os.path.join(args.cached_dir, "images_stats_info.parquet")
    if os.path.exists(file_path) and not args.refresh:
        print(f"{file_path} exists, skip generates meta info")
        return False

    # process
    list_all_images: List[str] = list(glob(os.path.join(args.data_dir, "*", "*.jpg")))
    df_train = pd.DataFrame({"file_path": list_all_images})
    df_train["image"] = df_train["file_path"].apply(lambda x: os.path.basename(x))
    df_train["kind"] = df_train["file_path"].apply(lambda x: os.path.split(os.path.dirname(x))[-1])
    with Pool(processes=args.n_jobs) as p:
        func = partial(process_image, functions=list_of_functions)
        df = pd.concat(list(p.map(func, list_all_images)), axis=1).T
        df.columns = [f"{i}_{m}" for m, i in df.columns]

    df["image"] = df_train["image"].tolist()
    df["kind"] = df_train["kind"].tolist()
    df.sort_values("image", inplace=True)

    df.to_parquet(file_path)
    print(f"{df.describe().T}")
    return


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
    #
    parser.add_argument('--gpus', default=None)
    # debug
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    args = parser.parse_args()

    # house keeping
    safe_mkdir(args.output_dir)
    safe_mkdir(args.cached_dir)
    safe_mkdir(args.model_dir)
    # start program
    main(args)
