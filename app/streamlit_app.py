"""Demo nhận diện biển báo: tải ảnh hoặc video lên và xem kết quả.

Chạy:  streamlit run app/streamlit_app.py
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from vnsign.config import ROOT, load_config, pick_device

st.set_page_config(page_title="Nhận diện biển báo VN", page_icon="🚦", layout="wide")
cfg = load_config()


@st.cache_resource
def get_model(path: str):
    from ultralytics import YOLO

    return YOLO(path)


st.title("🚦 Nhận diện biển báo giao thông Việt Nam")

found = sorted(ROOT.glob("runs/**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
with st.sidebar:
    st.header("Cài đặt")
    custom = st.text_input("Đường dẫn weights (để trống = tự chọn)", "")
    options = [str(p) for p in found]
    chosen = custom or (st.selectbox("Weights có sẵn", options) if options else "")
    conf = st.slider("Ngưỡng confidence", 0.05, 0.95, float(cfg["infer"]["conf"]), 0.05)

if not chosen:
    st.warning("Chưa có weights. Hãy train mô hình trước (runs/.../weights/best.pt).")
    st.stop()

model = get_model(chosen)
device = pick_device(cfg["train"]["device"])
imgsz = cfg["infer"]["imgsz"]
tab_img, tab_vid = st.tabs(["🖼️ Ảnh", "🎬 Video"])

with tab_img:
    up = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"], key="img")
    if up:
        img = Image.open(up).convert("RGB")
        r = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        c1, c2 = st.columns(2)
        c1.image(img, caption="Ảnh gốc", use_container_width=True)
        c2.image(r.plot()[..., ::-1], caption="Kết quả", use_container_width=True)
        if len(r.boxes):
            st.dataframe(pd.DataFrame({
                "biển báo": [r.names[int(c)] for c in r.boxes.cls],
                "độ tin cậy": [round(float(x), 3) for x in r.boxes.conf],
            }), use_container_width=True)
        else:
            st.info("Không phát hiện biển báo nào ở ngưỡng này.")

with tab_vid:
    upv = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if upv and st.button("Bắt đầu xử lý"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upv.name).suffix) as tmp:
            tmp.write(upv.read())
        frame_box, bar, seen = st.empty(), st.progress(0.0), {}
        stream = model.track(tmp.name, stream=True, persist=True, conf=conf, imgsz=imgsz,
                             device=device, tracker=cfg["infer"]["tracker"], verbose=False)
        for i, r in enumerate(stream):
            frame_box.image(r.plot()[..., ::-1], use_container_width=True)
            if r.boxes is not None and r.boxes.id is not None:
                for tid, c in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
                    seen.setdefault(tid, {"track_id": tid, "biển báo": r.names[c], "frame đầu": i})
            bar.progress(min(1.0, (i + 1) / 1000))  # tiến độ gần đúng
        bar.progress(1.0)
        st.success(f"Xong. Phát hiện {len(seen)} biển (đã khử trùng lặp).")
        if seen:
            st.dataframe(pd.DataFrame(seen.values()), use_container_width=True)
