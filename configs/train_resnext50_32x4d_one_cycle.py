configs = {
    "num_workers": 8,
    "batch_size": 6,  # efficientnet-b2

    "accumulate_grad_batches": 4,

    "n_epochs": 25,
    "lr": 4e-6,

    "lr_scheduler": "OneCycleLR",
    "scheduler_params": {},

    "augmentations": [
        {"transform": "HorizontalFlip", "params": {"p": .5}, },
        {"transform": "VerticalFlip", "params": {"p": .5}, },
        {"transform": "RandomRotate90", "params": {"always_apply": False, "p": .5}, },
        {"transform": "InvertImg", "params": {"p": .5}, },
        {"transform": "Resize", "params": {"height": 512, "width": 512, "always_apply": True, "p": 1.}, },
        # {"transform": "ToFloat", "params": {"max_value": 255, "always_apply": True, "p": 1.}, },
        {"transform": "Normalize", "params": {"always_apply": True, "p": 1.}, },
        {"transform": "ToTensorV2", "params": {"always_apply": True, "p": 1.}, },
    ],
}

