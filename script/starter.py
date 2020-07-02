import os
import random
import time
import warnings
from datetime import datetime
from glob import glob
from typing import List, Optional, Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
# Class Balance "on fly" from @CatalystTeam
from catalyst.data.sampler import BalanceClassSampler
from efficientnet_pytorch import EfficientNet
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from torch import nn
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
        x_padding = np.linspace(fpr_arr[mask][-1], 1, 100)

        x = np.concatenate([fpr_arr[mask], x_padding])
        y = np.concatenate([tpr_arr[mask], [y_max] * len(x_padding)])
        return metrics.auc(x, y - y_min)  # normalize such that curve starts at y=0

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    submetrics = [compute_submetrics(
        y_min=a, y_max=b, fpr_arr=fpr, tpr_arr=tpr) for a, b in zip(tpr_thresholds[:-1], tpr_thresholds[1:])]
    competition_metric = (np.array(submetrics) * weights).sum() / normalization
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
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.confidence: float = 1.0 - smoothing
        self.smoothing: float = smoothing

    def forward(self, x, target):
        if not self.training:
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

        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.model = model
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.log(f'Fitter prepared. Device is {self.device}')
        self._configure_fitter()

    def _configure_fitter(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = self.config.SchedulerClass(self.optimizer, **self.config.scheduler_params)
        self.criterion = LabelSmoothing().to(self.device)
        return self

    def fit(self, train_loader, validation_loader):

        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
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

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler = self.config.SchedulerClass(self.optimizer, **self.config.scheduler_params)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.criterion = LabelSmoothing().to(self.device)

        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        # self._configure_fitter()
        return self

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class _BaseRetriever(Dataset):
    def __init__(self, image_names: List[str], transforms: Optional[Callable] = None):
        super().__init__()
        self.image_names: List[str] = image_names
        self.transforms: Optional[Callable] = transforms

    def _load_one_image(self, index: int):
        image_name = self.image_names[index]
        if not os.path.exists(image_name):
            raise ValueError(f"file image does not exist: {image_name}")

        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return len(self.image_names)


def onehot_target(size: int, target: int):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class DatasetRetriever(_BaseRetriever):

    def __init__(self, image_names: List[str], labels: List[str], transforms: Optional[Callable] = None):
        super().__init__(image_names=image_names, transforms=transforms)
        self.labels: List[str] = labels
        self.nr_class: int = len(set(labels))

    def __getitem__(self, index: int):
        image_name, image = self._load_one_image(index=index)
        return image, onehot_target(self.nr_class, self.labels[index])

    def get_labels(self):
        return self.labels


class SubmissionRetriever(_BaseRetriever):
    def __init__(self, image_names: List[str], transforms: Optional[Callable] = None):
        super().__init__(image_names=image_names, transforms=transforms)

    def __getitem__(self, index: int):
        image_name, image = self._load_one_image(index=index)
        return os.path.basename(image_name), image


# Config
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
    validation_scheduler = True  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------

    # Augmentations
    # --------------------
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(always_apply=False, p=0.5),
        # A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.5),
        # A.InvertImg(always_apply=False, p=0.5),
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(always_apply=True),
        ToTensorV2(p=1.0),
    ], p=1.0)


class TrainOneCycleConfigs:
    num_workers: int = 8
    batch_size: int = 14  # 16

    # -------------------
    verbose: bool = True
    verbose_step: int = 1
    # -------------------

    n_epochs: int = 10
    lr: float = 0.001

    # --------------------
    step_scheduler: bool = False  # do scheduler.step after optimizer.step
    validation_scheduler: bool = False
    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
        max_lr=0.001,
        epochs=n_epochs,
        steps_per_epoch=20000,  # int(len(train_dataset) / batch_size),
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=10 ** 5
    )
    # --------------------

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
    batch_size: int = 12  # 16

    transforms: A.Compose = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(always_apply=True),
        ToTensorV2(p=1.0),
    ], p=1.0)


class TestConfigs:
    num_workers = 8
    batch_size = 12  # 16

    transforms: A.Compose = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(always_apply=True),
        ToTensorV2(p=1.0),
    ], p=1.0)


def inference(dataset: Dataset, configs, model: nn.Module) -> pd.DataFrame:
    data_loader = DataLoader(
        dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers, drop_last=False, )

    model.eval()
    result = {'Id': [], 'Label': []}
    total_num_batch: int = int(len(dataset) / configs.batch_size)
    for step, (image_names, images) in enumerate(data_loader):
        print(f"Test Batch: {step:03d} / {total_num_batch:d}, {100. * step/total_num_batch: .02f}", end='\r')

        y_pred = model(images.cuda())
        y_pred = 1. - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]

        result['Id'].extend(image_names)
        result['Label'].extend(y_pred)

    submission = pd.DataFrame(result).sort_values("Id").set_index("Id")
    print(f"Finish Test: {submission.shape}: \n{submission.descirbe().T}")
    return submission


def main():
    seed_everything()

    working_dir: str = "../input/alaska2-image-steganalysis/"
    test_dir: str = "Test"

    labels: List[str] = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    label_map = {kind: label for label, kind in enumerate(labels)}
    training_images: List[str] = list(
        filter(lambda x: test_dir != os.path.split(os.path.dirname(x))[-1],
               glob(os.path.join(working_dir, "*", "*.jpg"))))
    df_train = pd.DataFrame({"file_path": training_images})
    df_train["image"] = df_train["file_path"].apply(lambda x: os.path.basename(x))
    df_train["algo"] = df_train["file_path"].apply(lambda x: os.path.split(os.path.dirname(x))[-1])
    df_train["label"] = df_train["algo"].map(label_map).astype(np.int32)
    print(f"{df_train.nunique()}")

    # create net
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)

    # checkpoint = torch.load("./best-checkpoint-033epoch.bin")
    # net.load_state_dict(checkpoint['model_state_dict'])

    model = net.cuda()
    device = torch.device('cuda:0')

    # if False:
    if True:  # training
        training_configs = TrainOneCycleConfigs
        validation_configs = ValidConfigs

        gkf = GroupKFold(n_splits=5)
        for train_index, vallid_index in gkf.split(X=df_train["label"], y=df_train["label"], groups=df_train["image"]):
            break

        train_df = df_train.iloc[train_index]
        valid_df = df_train.iloc[vallid_index]

        train_dataset = DatasetRetriever(
            image_names=train_df["file_path"].tolist(), labels=train_df["label"].tolist(),
            transforms=training_configs.transforms)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
            batch_size=training_configs.batch_size,
            pin_memory=False,
            drop_last=True,
            num_workers=training_configs.num_workers,
        )

        validation_dataset = DatasetRetriever(
            image_names=valid_df["file_path"].tolist(), labels=valid_df["label"].tolist(),
            transforms=training_configs.transforms)
        val_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=validation_configs.batch_size,
            num_workers=validation_configs.num_workers,
            shuffle=False,
            sampler=SequentialSampler(validation_dataset),
            pin_memory=False,
        )

        fitter = Fitter(model=model, device=device, config=training_configs)
        # fitter.load(f'{fitter.base_dir}/best-checkpoint-024epoch.bin')
        fitter.fit(train_loader, val_loader)

    # Test
    dataset = SubmissionRetriever(
        image_names=glob(os.path.join(working_dir, test_dir, "*.jpg")),
        transforms=TestConfigs.transforms,
    )

    submission = inference(dataset=dataset, configs=TestConfigs, model=model)
    submission.to_csv('submission.csv', index=True)
    print(submission.head())
    return


if "__main__" == __name__:
    main()
