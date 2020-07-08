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


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return


# competition metrics
def alaska_weighted_auc(
        y_true: np.array, y_valid: np.array, tpr_thresholds: List[float] = [0.0, 0.4, 1.0],
        weights: List[float] = [2, 1]):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    def compute_submetrics(y_min: float, y_max: float, fpr_arr: np.array, tpr_arr: np.array) -> float:
        mask = (y_min < tpr_arr) & (tpr_arr < y_max)

        if not len(fpr[mask]):
            return 0.

        x_padding = np.linspace(fpr_arr[mask][-1], 1, 100)

        x = np.concatenate([fpr_arr[mask], x_padding])
        y = np.concatenate([tpr_arr[mask], [y_max] * len(x_padding)])
        return metrics.auc(x, y - y_min)  # normalize such that curve starts at y=0

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    sub_metrics = [compute_submetrics(
        y_min=a, y_max=b, fpr_arr=fpr, tpr_arr=tpr) for a, b in zip(tpr_thresholds[:-1], tpr_thresholds[1:])]
    competition_metric = (np.array(sub_metrics) * weights).sum() / normalization
    return competition_metric


# Metrics
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RocAucMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score


# Label Smoothing
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.05, enable: bool = True):
        super().__init__()
        self.confidence: float = 1.0 - smoothing
        self.smoothing: float = smoothing
        self.enable: bool = enable

    def forward(self, x, target):
        if not self.enable:
            return torch.nn.functional.cross_entropy(x, target)

        x = x.float()
        target = target.float()
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = (log_probs * target).sum(-1)
        smooth_loss = log_probs.mean(dim=-1)
        return -(self.confidence * nll_loss + self.smoothing * smooth_loss).mean()


# Fitter
class Fitter:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = "./"
        self.log_path = os.path.join(self.base_dir, "log.txt")
        self.best_summary_loss = 10 ** 5

        self.model: nn.Module = model
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.criterion: Optional[nn.Module] = None

        self.log(f"Fitter prepared. Device is {self.device}")
        self._configure_fitter()

    def _configure_fitter(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = self.config.lr_scheduler(self.optimizer, **self.config.scheduler_params)
        self.criterion = self.config.loss.to(self.device)
        return self

    def fit(self, train_loader, validation_loader):

        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                timestamp = datetime.utcnow().isoformat()
                self.log(f"\n{timestamp}\nLR: {lr}")

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)

            self.log(
                f"[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: "
                f"{final_scores.avg:.5f}, time: {(time.time() - t):.5f}")
            self.save(os.path.join(self.base_dir, "last-checkpoint.bin"))

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)

            self.log(
                f"[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: "
                f"{final_scores.avg:.5f}, time: {(time.time() - t):.5f}")
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(os.path.join(self.base_dir, f"best-checkpoint-{self.epoch:03d}epoch.bin"))
                for path in sorted(glob(os.path.join(self.base_dir, "best-checkpoint-*epoch.bin")))[:-3]:
                    os.remove(path)

            if self.config.step_after_validation:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if step % self.config.verbose_step == 0 and self.config.verbose:
                print(
                    f"Val Step {step}/{len(val_loader)}, summary_loss: {summary_loss.avg:.5f}, final_score: "
                    f"{final_scores.avg:.5f}, time: {(time.time() - t):.5f}",
                    end="\r"
                )

            with torch.no_grad():
                targets = targets.to(self.device).float()
                images = images.to(self.device).float()
                batch_size = images.shape[0]
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                #
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if step % self.config.verbose_step == 0 and self.config.verbose:
                print(
                    f"Train Step {step}/{len(train_loader)}, summary_loss: {summary_loss.avg:.5f}, final_score: "
                    f"{final_scores.avg:.5f}, time: {(time.time() - t):.5f}",
                        end="\r"
                    )

            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            #
            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()
            if self.config.step_after_optimizer:
                self.scheduler.step()

        return summary_loss, final_scores

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)
        return self

    def load(self, path, model_weights_only: bool = False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = self.config.lr_scheduler(self.optimizer, **self.config.scheduler_params)
        if not model_weights_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_summary_loss = checkpoint['best_summary_loss']
            self.epoch = checkpoint['epoch'] + 1

        return self

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class BaseLightningModule(pl.LightningModule):
    def __init__(
            self, model: nn.Module,
            training_records: Optional[List[Dict[str, Any]]] = None, training_configs: Optional = None,
            valid_records: Optional[List[Dict[str, Any]]] = None, valid_configs: Optional = None,
            eval_metric_name: str = "val_metric_score", eval_metric_func: Optional[Callable] = None, ):
        super().__init__()
        self.model: nn.Module = model
        # configs, records
        self.training_records: Optional[List[Dict[str, Any]]] = training_records
        self.training_configs = training_configs
        self.valid_records: Optional[List[Dict[str, Any]]] = valid_records
        self.valid_configs = valid_configs
        #
        self.restored_checkpoint = None
        #
        self.current_epoch: int = 0

        # eval metric
        self.eval_metric_name: str = eval_metric_name
        self.eval_metric_func: Optional[Callable] = eval_metric_func
        self.loss: Optional[nn.Module] = None
        if self.training_configs is not None:
            self.loss = self.training_configs.loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return {"loss": self.loss(y_hat, y), "y": y, "yhat": y_hat}

    def _post_process_outputs_for_metric(self, outputs):
        # metric
        y_true = (torch.cat([x["y"] for x in outputs], dim=0).cpu().numpy()[:, 0] == 0).astype(int)
        y_pred = 1. - F.softmax(torch.cat([x["yhat"] for x in outputs], dim=0)).data.cpu().numpy()[:, 0]
        return self.eval_metric_func(y_true, y_pred)

    def training_epoch_end(self, outputs) -> Dict[str, float]:
        tr_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        tr_metric = self._post_process_outputs_for_metric(outputs)
        metric_name: str = f"tr_{self.eval_metric_name}"
        print(f"loss: {tr_loss_mean:.6f}, {metric_name}: {tr_metric:.6f}\n")

        return {"loss": tr_loss_mean, metric_name: tr_metric}

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        x, y = batch
        y_hat = self.model(x)
        return {"val_loss": self.loss(y_hat, y), "y": y, "yhat": y_hat}

    def validation_epoch_end(self, outputs) -> Dict[str, float]:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        val_metric = self._post_process_outputs_for_metric(outputs)
        metric_name: str = f"val_{self.eval_metric_name}"
        print(f"\nval_loss: {val_loss_mean:.6f}, {metric_name}: {val_metric:.6f}")

        logs = {"val_loss": val_loss_mean, metric_name: val_metric}
        return {"val_loss": val_loss_mean, metric_name: val_metric, "log": logs}

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        train_dataset = DatasetRetriever(
            records=self.training_records, transforms=self.training_configs.transforms)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
            batch_size=self.training_configs.batch_size, pin_memory=True, drop_last=True,
            num_workers=self.training_configs.num_workers,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        validation_dataset = DatasetRetriever(
            records=self.valid_records, transforms=self.valid_configs.transforms)
        val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=self.valid_configs.batch_size, pin_memory=True, shuffle=False,
            num_workers=self.valid_configs.num_workers, sampler=SequentialSampler(validation_dataset), )
        return val_loader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(
            self) -> Union[optim.Optimizer, Sequence[optim.Optimizer], Dict, Sequence[Dict], Tuple[List, List], None]:
        optimizer = optim.AdamW(self.model.parameters(), lr=self.training_configs.lr)

        scheduler_params = self.training_configs.scheduler_params.copy()
        if "steps_per_epoch" in scheduler_params.keys():
            steps_per_epoch = int(len(self.training_records) // self.training_configs.batch_size) + 1
            scheduler_params.update({"steps_per_epoch": steps_per_epoch})

        scheduler = self.training_configs.lr_scheduler(optimizer, **scheduler_params)

        # restore checkpoint
        if self.restored_checkpoint is not None:
            optimizer.load_state_dict(self.restored_checkpoint["optimizer_states"])
            scheduler.load_state_dict(self.restored_checkpoint["lr_schedulers"])
            self.current_epoch = self.restored_checkpoint["epoch"] + 1

        return [optimizer], [scheduler]


# DataSet
class _BaseRetriever(Dataset):
    def __init__(self, records: List[Dict[str, Any]], transforms: Optional[Callable] = None):
        super().__init__()
        self.records: List[Dict[str, Any]] = records
        self.transforms: Optional[Callable] = transforms

    def _load_one_image(self, index: int) -> np.array:
        image_info: int = self.records[index]
        image_path: str = image_info["file_path"]
        image_name: str = image_info["image"]
        image_kind: str = image_info["kind"]

        if not os.path.exists(image_path):
            raise ValueError(f"file image does not exist: {image_kind}, {image_name}")

        # full_path: work_dir + kind + image_name
        # full_path, kind, image_name,
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image

    def __len__(self) -> int:
        return len(self.records)


def onehot_target(size: int, target: int):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class DatasetRetriever(_BaseRetriever):
    def __init__(self, records: List[Dict[str, Any]], transforms: Optional[Callable] = None):
        super().__init__(records=records, transforms=transforms)
        self.labels: List[int] = [record.get("label") for record in self.records]
        self.nr_class: int = len(set(self.labels))

    def __getitem__(self, index: int):
        image = self._load_one_image(index=index)
        label = onehot_target(self.nr_class, self.labels[index])
        # TODO: make mixup working
        return image, label

    def get_labels(self) -> List[int]:
        return self.labels


class SubmissionRetriever(_BaseRetriever):
    def __init__(self, records: List[Dict[str, Any]], transforms: Optional[Callable] = None):
        super().__init__(records=records, transforms=transforms)

    def __getitem__(self, index: int) -> Tuple[str, np.array]:
        image = self._load_one_image(index=index)
        image_info: int = self.records[index]
        image_name: str = image_info["image"]
        image_kind: str = image_info["kind"]
        return image_kind, image_name, image


def inference(dataset: Dataset, configs, model: nn.Module) -> pd.DataFrame:
    data_loader = DataLoader(
        dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers, drop_last=False, )

    model.eval()
    result = {'Id': [], 'Label': []}
    total_num_batch: int = int(len(dataset) / configs.batch_size)
    for step, (image_kinds, image_names, images) in enumerate(data_loader):
        print(
            f"Test Batch: {step:03d} / {total_num_batch:d}, progress: {100. * step / total_num_batch: .02f} %",
            end='\r')

        y_pred = model(images.cuda())
        y_pred = 1. - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]

        result['Id'].extend(image_names)
        result['Label'].extend(y_pred)

    submission = pd.DataFrame(result).sort_values("Id").set_index("Id")
    print(f"\nFinish Test: {submission.shape}, Stats:\n{submission.describe()}\n")
    return submission


def index_train_test_images(args):
    working_dir: str = args.data_dir
    output_dir: str = args.cached_dir
    meta_dir: str = args.meta_dir
    shared_indices: List[str] = ["image", "kind"]
    file_path_image_quality: str = "image_quality.csv"

    args.file_path_train_images_info = os.path.join(output_dir, "train_images_info.parquet")
    args.file_path_test_images_info = os.path.join(output_dir, "test_images_info.parquet")
    if os.path.exists(args.file_path_train_images_info) and os.path.exists(args.file_path_test_images_info):
        if not args.refresh_cache:
            return

    file_path_image_quality = os.path.join(meta_dir, file_path_image_quality)
    df_quality: pd.DataFrame = pd.DataFrame()
    if os.path.exists(file_path_image_quality):
        df_quality = pd.read_csv(file_path_image_quality).set_index(shared_indices)
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
    df_train.set_index(shared_indices, inplace=True)

    if not df_quality.empty:
        df_train = df_train.join(df_quality["quality"])

    print(f"{df_train.nunique()}")
    df_train.sort_values("image", inplace=True)
    df_train.loc[df_train["label"].notnull()].to_parquet(args.file_path_train_images_info)
    df_train.loc[df_train["label"].isnull()].to_parquet(args.file_path_test_images_info)
    return


def process_images_to_records(args, df: pd.DataFrame) -> List[Dict[str, Any]]:
    df["file_path"] = df.apply(lambda x: os.path.join(args.data_dir, x["kind"], x["image"]), axis=1)
    return df.to_dict("record")


def initialize_configs(filename: str):
    if not os.path.exists(filename):
        raise ValueError("Spec file {spec_file} does not exist".format(spec_file=filename))

    module_name = filename.split(os.sep)[-1].replace('.', '')

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lr_schedulers = {
    "ReduceLROnPlateau": {
        "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        "step_after_optimizer": False,  # do scheduler.step after optimizer.step
        "step_after_validation": True,  # do scheduler.step after validation stage loss
        "params": {
            "mode": "min",
            "factor": 0.5,
            "patience": 1,
            "verbose": True,
            "threshold": 0.0001,
            "threshold_mode": "abs",
            "cooldown": 0,
            "min_lr": 1e-8,
            "eps": 1e-08
        },
    },

    "OneCycleLR": {
        "lr_scheduler": optim.lr_scheduler.OneCycleLR,
        "step_after_optimizer": True,  # do scheduler.step after optimizer.step
        "step_after_validation": False,  # do scheduler.step after validation stage loss
        "params": {
            "max_lr": 0.001,
            "epochs": 5,
            "steps_per_epoch": 30000,  # int(len(train_dataset) / batch_size),
            "pct_start": 0.1,
            "anneal_strategy": "cos",
            "final_div_factor": 10 ** 5
        },
    },
}

augment_methods = {
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "RandomRotate90": A.RandomRotate90,
    "RandomGridShuffle": A.RandomGridShuffle,
    "InvertImg": A.InvertImg,
    "Resize": A.Resize,
    "Normalize": A.Normalize,
    "ToTensorV2": ToTensorV2,
}


def transform_factory(item, possible_methods: Dict[str, Any] = augment_methods):
    obj = possible_methods.get(item["transform"])
    params = item["params"]
    return obj(**params)


class BaseConfigs:
    def __init__(self, file_path: str):
        self.configs: Dict[str, Any] = self._load_configs(file_path)

        # -------------------
        self.num_workers: int = self.configs.get("num_workers", 8)
        self.batch_size: int = self.configs.get("batch_size", 16)

        # display
        # -------------------
        self.verbose: bool = self.configs.get("verbose", True)
        self.verbose_step: int = self.configs.get("verbose_step", 1)

        # -------------------
        self.n_epochs: int = self.configs.get("n_epochs", 5)
        self.lr: float = self.configs.get("lr", 0.001)

        # --------------------
        self.loss: Optional[nn.Module] = None

        # config scheduler
        # --------------------
        if "lr_scheduler" in self.configs.keys():
            tmp = lr_schedulers.get(self.configs["lr_scheduler"], None)
            self.step_after_optimizer: bool = tmp.get(
                "step_after_optimizer", False)  # do scheduler.step after optimizer.step
            self.step_after_validation: bool = tmp.get(
                "step_after_validation", False)  # do scheduler.step after validation stage loss
            self.lr_scheduler = tmp.get("lr_scheduler", None)

            # scheduler params
            self.scheduler_params: Dict = tmp.get("params", dict()).copy()
            laod_scheduler_params = self.configs.get("scheduler_params", dict())
            if laod_scheduler_params:
                self.scheduler_params.update(laod_scheduler_params)

            if "max_lr" in self.scheduler_params.keys():
                self.scheduler_params["max_lr"] = self.lr

            if "epochs" in self.scheduler_params.keys():
                self.scheduler_params["epochs"] = self.n_epochs

        self.transforms: List = self._load_transforms(self.configs["augmentations"])
        self.tta_transforms: List = [self.transforms]
        if "test_time_augmentations" in self.configs.keys():
            self.tta_transforms = [self._load_transforms(tta) for tta in self.configs["test_time_augmentations"]]

    @staticmethod
    def _load_transforms(augmentations: List):
        return A.Compose([transform_factory(item) for item in augmentations])

    @staticmethod
    def _load_configs(file_path: str):
        return initialize_configs(file_path).configs

    @classmethod
    def from_file(cls, file_path: str):
        return cls(file_path=file_path)


# Configs Block Starts
########################################################################################################################
class TrainReduceOnPlateauConfigs:
    num_workers: int = 8
    batch_size: int = 12  # 16

    # -------------------
    verbose: bool = True
    verbose_step: int = 1

    # -------------------
    n_epochs: int = 5
    lr: float = 0.001

    # --------------------
    loss: nn.Module = LabelSmoothing(smoothing=.05)

    # --------------------
    step_after_optimizer: bool = False  # do scheduler.step after optimizer.step
    step_after_validation = True  # do scheduler.step after validation stage loss
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )

    # Augmentations
    # --------------------
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(always_apply=False, p=0.5),
        # A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.5),
        A.InvertImg(always_apply=False, p=0.5),
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(always_apply=True),
        ToTensorV2(p=1.0),
    ], p=1.0)


class TrainOneCycleConfigs:
    num_workers: int = 8
    batch_size: int = 14  # efficientnet-b2
    # batch_size: int = 11  # efficientnet-b3
    # batch_size: int = 8  # efficientnet-b4
    # batch_size: int = 6  # efficientnet-b5
    # batch_size: int = 4  # efficientnet-b6

    # -------------------
    verbose: bool = True
    verbose_step: int = 1

    # -------------------
    n_epochs: int = 40
    lr: float = 0.001

    # --------------------
    loss: nn.Module = LabelSmoothing(smoothing=.05)

    # --------------------
    step_after_optimizer: bool = True  # do scheduler.step after optimizer.step
    step_after_validation: bool = False
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
        max_lr=0.001,
        epochs=n_epochs,
        steps_per_epoch=30000,  # int(len(train_dataset) / batch_size),
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=10 ** 5
    )

    # Augmentations
    # --------------------
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(always_apply=False, p=0.5),
        # A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.5),
        A.InvertImg(always_apply=False, p=0.5),
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(always_apply=True),
        ToTensorV2(p=1.0),
    ], p=1.0)


class ValidConfigs:
    num_workers: int = 8
    batch_size: int = 16  # 16

    transforms: A.Compose = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(always_apply=True),
        ToTensorV2(p=1.0),
    ], p=1.0)


class TestConfigs:
    num_workers = 8
    batch_size = 8  # 16

    transforms: A.Compose = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(always_apply=True),
        ToTensorV2(p=1.0),
    ], p=1.0)


########################################################################################################################
# Configs Block Ends


def main(args):
    args.labels: List[str] = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    eval_metric_name: str = "weighted_auc"

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
        df_train = pd.read_parquet(args.file_path_train_images_info).reset_index()
        if args.debug:
            df_train = df_train.iloc[:200]

        df_train["label"] = df_train["label"].astype(np.int32)
        df = df_train.loc[(~df_train["image"].duplicated(keep="first"))]

        skf = StratifiedKFold(n_splits=5)
        for train_index, valid_index in skf.split(X=df["label"], y=df["quality"], groups=df["image"]):
            break

        train_df = df_train.loc[df_train["image"].isin(df["image"].iloc[train_index])]
        valid_df = df_train.loc[df_train["image"].isin(df["image"].iloc[valid_index])]
        #
        training_records = process_images_to_records(args, df=train_df)
        valid_records = process_images_to_records(args, df=valid_df)

    # use lightning
    if args.use_lightning:
        model = BaseLightningModule(
            model, training_configs=training_configs, training_records=training_records,
            valid_configs=validation_configs, valid_records=valid_records, eval_metric_name=eval_metric_name,
            eval_metric_func=alaska_weighted_auc)

        # load_from_checkpoint: not working
        if args.load_checkpoint and os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])

            print(checkpoint["lr_schedulers"])
            # not working if using
            # trainer = Trainer(resume_From_checkpoint=args.checkpoint_path)
            if not args.load_weights_only:
                model.restored_checkpoint = checkpoint

        # training
        if not args.inference_only:
            metric_name: str = f"val_{eval_metric_name}"
            file_path_checkpoint: str = os.path.join(
                args.model_dir, "__".join(["result", args.model_arch, "{epoch:03d}-{val_loss:.4f}"]))
            checkpoint_callback = ModelCheckpoint(
                filepath=file_path_checkpoint, save_top_k=3, verbose=True, monitor=metric_name, mode="max")

            early_stop_callback = EarlyStopping(
                monitor=metric_name, min_delta=0., patience=5, verbose=True, mode='max')
            trainer = Trainer(
                gpus=args.gpus, min_epochs=1, max_epochs=1000, default_root_dir=args.model_dir,
                # distributed_backend="dpp",
                early_stop_callback=early_stop_callback,
                checkpoint_callback=checkpoint_callback
            )

            trainer.fit(model)
            model.freeze()

    # raw
    model = model.cuda()
    device = torch.device('cuda:0')

    if not args.inference_only and not args.use_lightning:
        train_dataset = DatasetRetriever(records=training_records, transforms=training_configs.transforms)
        train_loader = torch.utils.data.DataLoader(
            DatasetRetriever(records=training_records, transforms=training_configs.transforms),
            sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
            batch_size=training_configs.batch_size,
            pin_memory=False,
            drop_last=True,
            num_workers=training_configs.num_workers,
        )
        validation_dataset = DatasetRetriever(records=valid_records, transforms=training_configs.transforms)
        val_loader = torch.utils.data.DataLoader(
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
        # fitter.load(f'{fitter.base_dir}/best-checkpoint-024epoch.bin')
        fitter.fit(train_loader, val_loader)

    # Test
    test_configs = BaseConfigs.from_file(file_path=args.test_configs)
    df_test = pd.read_parquet(args.file_path_test_images_info).reset_index()
    if args.tta:
        collect = list()
        for i, tta in enumerate(test_configs.tta_transforms, 1):
            print(f"TTA: {i:02d} / {len(test_configs.tta_transforms):02d} rounds")
            dataset = SubmissionRetriever(records=process_images_to_records(args, df=df_test), transforms=tta, )
            submission = inference(dataset=dataset, configs=test_configs, model=model)
            collect.append(submission)

        submission = pd.concat(collect, ).groupby(level=-1).mean()
    else:
        dataset = SubmissionRetriever(
            records=process_images_to_records(args, df=df_test), transforms=test_configs.transforms, )
        submission = inference(dataset=dataset, configs=test_configs, model=model)

    submission.to_csv('submission.csv', index=True)
    print(f"\nSubmission:\n{submission.head()}")
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
    default_model_arch: str = "efficientnet-b2"
    #
    default_train_configs: str = "../configs/train_baseline.py"
    default_valid_configs: str = "../configs/train_baseline.py"
    default_test_configs: str = "../configs/test_baseline.py"
    #
    default_n_jobs: int = 8
    default_init_seed: int = 42

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="folder for output")
    parser.add_argument("--cached-dir", type=str, default=default_cached_dir, help="folder for cached data")
    parser.add_argument("--meta-dir", type=str, default=default_meta_dir, help="folder for meta data")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="folder for models")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="folder for data")
    #
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
    parser.add_argument("--inference-only", action="store_true", default=False, help="only perform inference")
    parser.add_argument("--use-lightning", action="store_true", default=False, help="using lightning trainer")
    parser.add_argument("--refresh-cache", action="store_true", default=False, help="refresh cached data")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs, help="num worker")
    parser.add_argument("--init-seed", type=int, default=default_init_seed, help="initialize random seed")
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
