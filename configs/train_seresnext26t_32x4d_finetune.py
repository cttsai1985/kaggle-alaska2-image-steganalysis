configs = {
    "num_workers": 8,
    "batch_size": 12,  # efficientnet-b2

    "n_epochs": 25,
    # "lr": 4e-05,
    "lr": 4e-06,

    "accumulate_grad_batches": 2,
    "lr_scheduler": "ReduceLROnPlateau",
    "scheduler_params": {},

    "augmentations": [
        {"transform": "HorizontalFlip", "params": {"p": .5}, },
        {"transform": "VerticalFlip", "params": {"p": .5}, },
        {"transform": "RandomRotate90", "params": {"always_apply": False, "p": .5}, },
        {"transform": "InvertImg", "params": {"p": .5}, },
        {"transform": "Resize", "params": {"height": 512, "width": 512, "always_apply": True, "p": 1.}, },
        {"transform": "Normalize", "params": {"always_apply": True, "p": 1.}, },
        {"transform": "ToTensorV2", "params": {"always_apply": True, "p": 1.}, },
    ],
}

