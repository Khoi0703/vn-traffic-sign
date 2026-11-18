"""Huấn luyện YOLO11. Mọi tham số lấy từ configs/default.yaml (mục train)."""
from __future__ import annotations

import argparse
import sys

from vnsign.config import load_config, pick_device, resolve_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", help="vd. yolo11s.pt")
    ap.add_argument("--name", help="tên lần chạy (thư mục trong runs/)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--resume", help="đường dẫn last.pt để train tiếp")
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = load_config(a.config)
    t = dict(cfg["train"])
    for k in ("model", "name", "epochs", "imgsz", "batch", "device"):
        if getattr(a, k) is not None:
            t[k] = getattr(a, k)

    if a.resume:
        YOLO(a.resume).train(resume=True)
        return

    data_yaml = resolve_path(cfg["data"]["processed_dir"]) / "dataset.yaml"
    if not data_yaml.exists():
        sys.exit("Chưa có dataset. Hãy chạy 'Prepare data' (vnsign-prepare) trước.")

    device = pick_device(t.pop("device", "auto"))
    print(f"Thiết bị: {device}")
    model = YOLO(t.pop("model"))
    model.train(
        data=str(data_yaml),
        device=device,
        project=str(resolve_path(t.pop("project"))),
        exist_ok=True,
        **t,
    )


if __name__ == "__main__":
    main()
