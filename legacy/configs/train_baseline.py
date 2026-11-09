configs = {
    "num_workers": 8,
    "batch_size": 14,  # efficientnet-b2

    "n_epochs": 5,
    "lr": 8e-7,

    "lr_scheduler": "ReduceLROnPlateau",
    "scheduler_params": {},

    "augmentations": [
        # {"transform": "HorizontalFlip", "params": {"p": .5}, },
        # {"transform": "VerticalFlip", "params": {"p": .5}, },
        # {"transform": "RandomRotate90", "params": {"always_apply": False, "p": .5}, },
        # {"transform": "InvertImg", "params": {"p": .5}, },
        {"transform": "Resize", "params": {"height": 512, "width": 512, "always_apply": True, "p": 1.}, },
        # {"transform": "ToFloat", "params": {"max_value": 255, "always_apply": True, "p": 1.}, },
        {"transform": "Normalize", "params": {"always_apply": True, "p": 1.}, },
        {"transform": "ToTensorV2", "params": {"always_apply": True, "p": 1.}, },
    ],
}

