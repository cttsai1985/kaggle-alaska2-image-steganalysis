import os
import random
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import torch

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


def safe_mkdir(directory: str) -> bool:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"make dir: {directory}")
        return True

    print(f"skip making dir: {directory}")
    return False


def initialize_configs(filename: str):
    if not os.path.exists(filename):
        raise ValueError("Spec file {spec_file} does not exist".format(spec_file=filename))

    module_name = filename.split(os.sep)[-1].replace(".", "")

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return
