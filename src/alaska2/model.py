"""One timm backbone with a 4-class head; pretrained weights are pinned by hub revision and sha256."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from alaska2.constants import CLASSES
from alaska2.identity import sha256_file


class WeightsIdentityError(RuntimeError):
    pass


def resolve_pretrained(mcfg: dict) -> dict:
    """Download (or reuse the hub cache for) the pinned weight file and verify its sha256."""
    from huggingface_hub import hf_hub_download

    revision = mcfg.get("weights_revision")
    if not revision:
        raise WeightsIdentityError("model.weights_revision must pin a hub commit when pretrained = true")
    path = hf_hub_download(mcfg["weights_repo"], mcfg["weights_file"], revision=revision)
    got = sha256_file(path)
    if mcfg.get("weights_sha256") and got != mcfg["weights_sha256"]:
        raise WeightsIdentityError(f"pretrained weights sha256 {got} != pinned {mcfg['weights_sha256']}")
    return {
        "repo": mcfg["weights_repo"],
        "file": mcfg["weights_file"],
        "revision": revision,
        "sha256": got,
        "local_path": path,
    }


def build_model(cfg: dict, pretrained: bool | None = None) -> tuple[nn.Module, dict]:
    """Returns (model, weights_identity). ``pretrained=False`` is used when a checkpoint is loaded next."""
    import timm

    mcfg = cfg["model"]
    use_pretrained = mcfg["pretrained"] if pretrained is None else pretrained
    kwargs = {"num_classes": len(CLASSES), "drop_rate": float(mcfg.get("drop_rate", 0.0))}
    if use_pretrained:
        ident = resolve_pretrained(mcfg)
        model = timm.create_model(
            mcfg["backbone"], pretrained=True, pretrained_cfg_overlay={"file": ident["local_path"]}, **kwargs
        )
        ident = {k: v for k, v in ident.items() if k != "local_path"}
    else:
        model = timm.create_model(mcfg["backbone"], pretrained=False, **kwargs)
        ident = {"repo": None, "revision": None, "sha256": None, "note": "random init or checkpoint load"}
    pcfg = getattr(model, "pretrained_cfg", {}) or {}
    ident["timm_pretrained_mean"] = list(pcfg.get("mean", ()))
    ident["timm_pretrained_std"] = list(pcfg.get("std", ()))
    return model, ident


def state_dict_sha256(model_or_sd: nn.Module | dict) -> str:
    """Content hash of weights: sorted keys, dtype, shape and raw bytes (independent of pickle layout)."""
    sd = model_or_sd.state_dict() if isinstance(model_or_sd, nn.Module) else model_or_sd
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().to("cpu").contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    # weights_only=True: the checkpoint holds tensors and plain Python containers only.
    return torch.load(path, map_location=map_location, weights_only=True)
