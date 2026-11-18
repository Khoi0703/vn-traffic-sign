"""Nhận diện biển báo trên ảnh / thư mục ảnh / video / webcam.

Với video và webcam: dùng ByteTrack để mỗi biển chỉ được ghi 1 lần,
kết quả lưu ra video đã vẽ khung và file signs_log.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vnsign.config import load_config, pick_device, resolve_path
from vnsign.data.source import IMG_EXTS
from vnsign.eval.evaluate import find_latest_weights


def run(weights, source, conf=0.35, imgsz=640, device="auto", show=False,
        out_dir=None, tracker="bytetrack.yaml"):
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    device = pick_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = int(source) if str(source).isdigit() else str(source)  # "0" = webcam
    if isinstance(src, str) and (Path(src).is_dir() or Path(src).suffix.lower() in IMG_EXTS):
        model.predict(src, conf=conf, imgsz=imgsz, device=device, save=True,
                      project=str(out_dir.parent), name=out_dir.name, exist_ok=True)
        print(f"Đã lưu ảnh kết quả vào {out_dir}")
        return

    fps = 30.0
    if isinstance(src, str):
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    seen, writer = {}, None
    stream = model.track(src, stream=True, persist=True, conf=conf, imgsz=imgsz,
                         device=device, tracker=tracker, verbose=False)
    try:
        for i, r in enumerate(stream):
            frame = r.plot()
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                for tid, c, cf in zip(ids, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    rec = seen.get(tid)
                    if rec is None:
                        seen[tid] = {"track_id": tid, "class_id": c, "class_name": r.names[c],
                                     "first_frame": i, "first_time_s": round(i / fps, 2),
                                     "best_conf": round(cf, 3)}
                    elif cf > rec["best_conf"]:
                        rec.update(best_conf=round(cf, 3), class_id=c, class_name=r.names[c])
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            if show:
                cv2.imshow("vnsign (nhấn q để thoát)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    log = out_dir / "signs_log.csv"
    with open(log, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "class_id", "class_name",
                                          "first_frame", "first_time_s", "best_conf"])
        w.writeheader()
        w.writerows(seen.values())
    print(f"Phát hiện {len(seen)} biển (đã khử trùng lặp bằng tracking).")
    print(f"Video: {out_dir / 'annotated.mp4'} | Log: {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights")
    ap.add_argument("--source", required=True, help="ảnh | thư mục | video | 0 (webcam)")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ xem trực tiếp")
    ap.add_argument("--out", help="thư mục lưu kết quả")
    a = ap.parse_args()

    cfg = load_config(a.config)
    weights = a.weights or find_latest_weights(resolve_path(cfg["train"]["project"]))
    assert weights, "Không tìm thấy best.pt. Hãy train trước hoặc truyền --weights."
    out = a.out or resolve_path(cfg["paths"]["reports_dir"]) / "predict"
    run(weights, a.source, conf=a.conf or cfg["infer"]["conf"], imgsz=cfg["infer"]["imgsz"],
        device=cfg["train"]["device"], show=a.show, out_dir=out, tracker=cfg["infer"]["tracker"])


if __name__ == "__main__":
    main()
