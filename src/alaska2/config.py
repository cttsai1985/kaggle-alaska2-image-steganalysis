"""Versioned TOML configs with single-parent ``extends`` and a canonical content hash."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from alaska2.constants import CLASSES, PURPOSE_LABELS

SCHEMA_VERSION = 1


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """``a.b=value`` where value is parsed as a TOML value, falling back to a bare string."""
    key, _, raw = text.partition("=")
    if not key or not _:
        raise ValueError(f"override must be key=value: {text!r}")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    return key.strip(), value


def _load_raw(path: Path) -> dict:
    raw = tomllib.loads(path.read_text())
    parent = raw.pop("extends", None)
    return _deep_merge(_load_raw(path.parent / parent), raw) if parent else raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = _load_raw(path)
    for item in overrides or []:
        k, v = parse_override(item)
        _set_dotted(cfg, k, v)
    validate_config(cfg)
    cfg["_source"] = {"path": str(path), "overrides": list(overrides or [])}
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if cfg.get("purpose") not in PURPOSE_LABELS:
        raise ValueError(f"purpose must be one of {sorted(PURPOSE_LABELS)}")
    if tuple(cfg["data"]["classes"]) != CLASSES:
        raise ValueError(f"class order must be {CLASSES} (legacy order; score = 1 - P(Cover))")
    if cfg["data"]["decoder"] not in ("opencv", "pillow"):
        raise ValueError("data.decoder must be opencv or pillow")
    if cfg["train"]["device"] not in ("cpu", "cuda"):
        raise ValueError("train.device must be cpu or cuda (explicit; no silent fallback)")
    if cfg["train"]["amp_dtype"] not in ("bf16", "fp16", "none"):
        raise ValueError("train.amp_dtype must be bf16, fp16 or none")
    if not 0 <= int(cfg["train"]["num_workers"]) <= 16:
        raise ValueError("train.num_workers must be within 0..16 (operator cap)")
    for t in cfg["transforms"]["train"] + cfg["transforms"]["valid"]:
        if t["op"] not in ("hflip", "vflip", "rot90"):
            raise ValueError(f"transform {t['op']!r} is not pixel-preserving; only hflip/vflip/rot90 allowed")


def config_hash(cfg: dict) -> str:
    """sha256 of the canonical JSON of the resolved config, excluding the ``_source`` bookkeeping key."""
    body = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def data_paths(cfg: dict) -> dict[str, Path]:
    d = cfg["data"]
    root = Path(d["root"]).expanduser()
    return {
        "root": root,
        "images_dir": root / d["images_dir"],
        "manifest": root / d["manifest"],
        "folds": root / d["folds"],
        "runs_dir": root / d.get("runs_dir", "runs"),
    }
