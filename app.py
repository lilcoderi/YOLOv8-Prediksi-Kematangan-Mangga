import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import numpy as np

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
/* Container utama selectbox */
div[data-baseweb="select"] > div {
    background-color: white !important;   /* kotak dropdown putih */
    border-radius: 6px;
    border: 1px solid #ccc;
}

/* Teks yang sedang dipilih di dalam selectbox */
div[data-baseweb="select"] > div > div {
    color: black !important;              /* teks item terpilih hitam */
    font-weight: 600;
}

/* Saat membuka dropdown: item-itemnya */
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
    uploaded_file = st.file_uploader("Pilih file gambar (jpg/jpeg/png):", type=["jpg", "jpeg", "png"])
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
    camera_image = st.camera_input("Klik tombol di bawah untuk mengambil gambar:")
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
# 🎥 3. WEBCAM / REAL-TIME
# ============================
elif input_method == "Webcam / Kamera HP (Real-Time)":
    st.subheader("🎥 Real-Time Webcam")
    st.markdown("**Aktifkan webcam untuk mendeteksi secara real-time.**")
    run_webcam = st.checkbox("▶️ Jalankan Webcam")

    frame_display = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("🚫 Kamera tidak tersedia.")
        else:
            st.success("🟢 Webcam aktif. Hilangkan centang untuk menghentikan.")
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal membaca frame dari kamera.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)

                for r in results:
                    detected_frame = r.plot()

                frame_display.image(detected_frame, channels="BGR")

            cap.release()
