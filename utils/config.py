"""YAML loading, merging, and a few small config helpers."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_update(base: dict, extra: dict) -> dict:
    """Recursively merge ``extra`` onto a copy of ``base``. Lists are replaced."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def load_config(path: str | Path, root: str | Path | None = None) -> dict:
    """Load ``configs/default.yaml`` and overlay the requested file.

    If ``path`` already points at the default file, the overlay is a no-op.
    """
    path = Path(path)
    root = Path(root) if root is not None else _find_root(path)
    default_path = root / "configs" / "default.yaml"
    if not default_path.is_file():
        raise FileNotFoundError(f"missing {default_path}")
    cfg = load_yaml(default_path)
    if path.resolve() != default_path.resolve():
        cfg = deep_update(cfg, load_yaml(path))
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(root)
    return cfg


def _find_root(path: Path) -> Path:
    """Walk parents until a ``configs/default.yaml`` is found."""
    start = path if path.is_dir() else path.parent
    for candidate in [start, *start.parents]:
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    # Fall back to the current working directory, then the repo layout.
    cwd = Path.cwd()
    if (cwd / "configs" / "default.yaml").is_file():
        return cwd
    raise FileNotFoundError("Could not locate configs/default.yaml")


def resolve_device(name: str | None) -> str:
    import torch

    if name in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(name)
