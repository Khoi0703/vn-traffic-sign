"""Đọc cấu hình và các tiện ích dùng chung."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Đường dẫn tương đối được hiểu là tính từ thư mục gốc project."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def pick_device(setting: str = "auto") -> str:
    """auto -> GPU NVIDIA ('0') nếu có, rồi Apple 'mps', còn lại 'cpu'."""
    if setting != "auto":
        return str(setting)
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
