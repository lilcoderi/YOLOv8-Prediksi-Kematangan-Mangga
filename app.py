import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase  # <-- IMPORT diletakkan di atas

# ============================
# 🎨 PAGE CONFIG & CUSTOM STYLE
# ============================
st.set_page_config(page_title="Deteksi Kematangan Mangga", page_icon="🥭", layout="wide")

# CSS CUSTOM
st.markdown("""
<style>
/* Background utama */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right top, #fdfbfb, #ebedee);
    color: black;
}

/* Sidebar dengan gradasi dan teks hitam */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffe259, #ffa751);
    color: black;
}
[data-testid="stSidebar"] * {
    color: black !important;
}

/* Card untuk title */
.title-card {
    background: linear-gradient(180deg, #ffe259, #ffa751);
    padding: 12px;
    border-radius: 10px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.15);
    margin-bottom: 15px;
    text-align: center;
}

/* H1 di dalam card (perbesar & capslock) */
.title-card h1 {
    color: black;
    font-size: 1.5rem;
    font-weight: 800;
    text-transform: uppercase;
    margin: 0;
}

/* =====================================
   STYLING UNTUK SELECTBOX (dropdown)
   ===================================== */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border-radius: 6px;
    border: 1px solid #ccc;
}

div[data-baseweb="select"] > div > div {
    color: black !important;
    font-weight: 600;
}

ul[role="listbox"] {
    background-color: white !important;
    color: black !important;
}
ul[role="listbox"] > li {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ============================
# 🎯 APP CONTENT
# ============================

# Title dengan card
st.markdown('<div class="title-card"><h1>🥭 DETEKSI KEMATANGAN MANGGA - YOLOV8</h1></div>', unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.write("Atur model dan sumber gambar di sini:")

# === PILIH MODEL ===
model_options = {
    "🌎 Global": "runs_manual/train/mangga_yolov8_manual/weights/best.pt",
    "✨ YoloV8 (Type l)": "runs_roboflow/mangga_yolov8_roboflow/weights/best.pt",
    "🍃 YoloV8 (Type n)": "runs_roboflow/mangga_yolov8_roboflow2/weights/best.pt"
}
selected_model = st.sidebar.selectbox("🔍 Pilih Model Deteksi", list(model_options.keys()))
model_path = model_options[selected_model]
model = YOLO(model_path)

# === PILIH METODE INPUT ===
input_method = st.sidebar.radio(
    "📷 Pilih Sumber Gambar",
    ["Upload JPG/IMG", "Upload dari Kamera", "Webcam / Kamera HP (Real-Time)"]
)

# ============================
# 📂 1. UPLOAD FILE GAMBAR
# ============================
if input_method == "Upload JPG/IMG":
    st.subheader("📂 Unggah Gambar Mangga")
    st.markdown("**Pilih file gambar (jpg/jpeg/png):**")
    uploaded_file = st.file_uploader("Silakan pilih", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        col1, col2 = st.columns(2)
        with col1:
            st.image(temp_path, caption="📸 Gambar yang Diunggah", use_column_width=True)

        results = model(temp_path)
        for r in results:
            boxes = r.plot()
            with col2:
                st.image(boxes, caption="✅ Hasil Deteksi", use_column_width=True)

        os.remove(temp_path)

# ============================
# 📷 2. UPLOAD DARI KAMERA
# ============================
elif input_method == "Upload dari Kamera":
    st.subheader("📸 Ambil Gambar dari Kamera")
    st.markdown("**Klik tombol di bawah untuk mengambil gambar:**")
    camera_image = st.camera_input("Silakan klik")
    if camera_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(camera_image.read())
            temp_path = temp_file.name

        col1, col2 = st.columns(2)
        with col1:
            st.image(temp_path, caption="📸 Gambar dari Kamera", use_column_width=True)

        results = model(temp_path)
        for r in results:
            boxes = r.plot()
            with col2:
                st.image(boxes, caption="✅ Hasil Deteksi", use_column_width=True)

        os.remove(temp_path)

# ============================
# 🎥 3. WEBCAM / REAL-TIME (HP & PC via browser)
# ============================
elif input_method == "Webcam / Kamera HP (Real-Time)":
    st.subheader("🎥 Real-Time Webcam")
    st.markdown("**Arahkan kamera Anda ke mangga untuk deteksi real-time.**")

    # Kelas untuk memproses frame video
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            # Konversi frame ke array BGR
            img = frame.to_ndarray(format="bgr24")
            # Jalankan deteksi menggunakan model YOLO
            results = model(img)
            for r in results:
                img = r.plot()
            return img

    # Menjalankan streaming video
    webrtc_streamer(
        key="mangga-detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
