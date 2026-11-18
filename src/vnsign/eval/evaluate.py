"""Đánh giá mô hình: chỉ số tổng, AP theo từng lớp (CSV) và confusion matrix."""
from __future__ import annotations

import argparse
from collections import Counter

from vnsign.config import load_config, pick_device, resolve_path


def find_latest_weights(runs_dir):
    cands = list(runs_dir.glob("**/weights/best.pt"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", help="mặc định: best.pt mới nhất trong runs/")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--imgsz", type=int)
    a = ap.parse_args()

    import pandas as pd
    from ultralytics import YOLO

    cfg = load_config(a.config)
    weights = a.weights or find_latest_weights(resolve_path(cfg["train"]["project"]))
    assert weights, "Không tìm thấy best.pt. Hãy train trước hoặc truyền --weights."
    print(f"Weights: {weights}")

    proc = resolve_path(cfg["data"]["processed_dir"])
    reports = resolve_path(cfg["paths"]["reports_dir"])
    split_key = "val" if a.split == "valid" else "test"

    model = YOLO(str(weights))
    m = model.val(
        data=str(proc / "dataset.yaml"),
        split=split_key,
        imgsz=a.imgsz or cfg["train"]["imgsz"],
        device=pick_device(cfg["train"]["device"]),
        project=str(reports),
        name=f"eval_{a.split}",
        exist_ok=True,
        plots=True,
    )
    b = m.box
    print(f"\n[{a.split}] mAP50={b.map50:.4f} | mAP50-95={b.map:.4f} | P={b.mp:.4f} | R={b.mr:.4f}")

    n_inst = Counter()
    for f in (proc / a.split / "labels").glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                n_inst[int(float(line.split()[0]))] += 1

    rows = [
        {
            "class_id": int(c),
            "class": m.names[int(c)],
            "instances": n_inst.get(int(c), 0),
            "precision": float(b.p[i]),
            "recall": float(b.r[i]),
            "AP50": float(b.ap50[i]),
            "AP50-95": float(b.ap[i]),
        }
        for i, c in enumerate(b.ap_class_index)
    ]
    df = pd.DataFrame(rows).sort_values("AP50")
    csv_path = reports / f"per_class_{a.split}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\n10 lớp yếu nhất (AP50 thấp nhất):")
    print(df.head(10).to_string(index=False))
    print(f"\nĐã lưu: {csv_path}")


if __name__ == "__main__":
    main()
