"""Export mô hình sang ONNX / OpenVINO và đo tốc độ (FPS) so với bản .pt."""
from __future__ import annotations

import argparse
import time

from vnsign.config import load_config, resolve_path
from vnsign.data.source import IMG_EXTS
from vnsign.eval.evaluate import find_latest_weights


def benchmark(model, images, imgsz, warmup=3) -> float:
    for im in images[:warmup]:
        model.predict(im, imgsz=imgsz, device="cpu", verbose=False)
    t0 = time.perf_counter()
    for im in images:
        model.predict(im, imgsz=imgsz, device="cpu", verbose=False)
    return len(images) / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights")
    ap.add_argument("--formats", nargs="+", default=["onnx", "openvino"])
    ap.add_argument("--int8", action="store_true", help="lượng tử hoá INT8 (chỉ openvino)")
    ap.add_argument("--n", type=int, default=50, help="số ảnh test dùng để benchmark")
    a = ap.parse_args()

    import pandas as pd
    from ultralytics import YOLO

    cfg = load_config(a.config)
    imgsz = cfg["train"]["imgsz"]
    weights = a.weights or find_latest_weights(resolve_path(cfg["train"]["project"]))
    assert weights, "Không tìm thấy best.pt."
    proc = resolve_path(cfg["data"]["processed_dir"])
    imgs = sorted(
        str(p) for p in (proc / "test" / "images").iterdir() if p.suffix.lower() in IMG_EXTS
    )[: a.n]
    assert imgs, "Chưa có ảnh test. Hãy chạy prepare."

    results = [{"format": "pytorch", "FPS (CPU)": benchmark(YOLO(str(weights)), imgs, imgsz)}]
    for fmt in a.formats:
        kw = {"int8": True, "data": str(proc / "dataset.yaml")} if a.int8 and fmt == "openvino" else {}
        out = YOLO(str(weights)).export(format=fmt, imgsz=imgsz, **kw)
        label = fmt + ("-int8" if kw else "")
        results.append(
            {"format": label, "FPS (CPU)": benchmark(YOLO(out, task="detect"), imgs, imgsz)}
        )
        print(f"Đã export {label}: {out}")

    df = pd.DataFrame(results).round(2)
    print("\n", df.to_string(index=False))
    reports = resolve_path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports / "benchmark.csv", index=False)


if __name__ == "__main__":
    main()
