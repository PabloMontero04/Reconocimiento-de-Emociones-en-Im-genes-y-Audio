import streamlit as st
import os
import sys
import time
import hashlib
from PIL import Image

# ────────────────────  Rutas de proyecto  ────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.detect_emotion_image  import predict_emotion_image
from src.predict_emotion_audio import predict_emotion_audio   # ← nuevo

# ────────────────────  Config Streamlit  ─────────────────────
st.set_page_config(page_title="Detector de Emociones", layout="centered")
st.title("🧠 Detector de Emociones en Imagen y Audio")

# Carpetas
IMG_FOLDER  = "uploaded_images"
AUDIO_FOLDER = "uploaded_audio"
os.makedirs(IMG_FOLDER,   exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Utilidad de hash para detección de cambios
def md5(b): return hashlib.md5(b).hexdigest()

# Estado de sesión para controlar recargas
st.session_state.setdefault("last_img_hash", None)
st.session_state.setdefault("last_audio_hash", None)

# ────────────────────  Interfaz con tabs  ────────────────────
tab_img, tab_audio = st.tabs(["🖼 Imagen", "🎙 Audio"])

# ========= TAB IMAGEN =========
with tab_img:
    st.markdown("### Sube una imagen con rostro para detectar la emoción.")
    img_file = st.file_uploader("📷 Imagen (JPG / PNG)", type=["jpg", "jpeg", "png"], key="img_uploader")

    temp_img = "temp.jpg"

    if img_file:
        img_bytes  = img_file.read()
        img_hash   = md5(img_bytes)

        if st.session_state.last_img_hash != img_hash:
            st.session_state.last_img_hash = img_hash

            ts = int(time.time())
            ext = os.path.splitext(img_file.name)[1]
            saved_path = os.path.join(IMG_FOLDER, f"img_{ts}{ext}")

            with open(saved_path, "wb") as f: f.write(img_bytes)
            with open(temp_img,  "wb") as f: f.write(img_bytes)

            st.rerun()

    if os.path.exists(temp_img):
        st.image(Image.open(temp_img), caption="Imagen subida", use_container_width=True)

        with st.spinner("Detectando emoción..."):
            emo, _, probs = predict_emotion_image(temp_img)

        st.success(f"🎯 Emoción detectada: **{emo.upper()}**")

        if probs:
            st.markdown("#### 📊 Probabilidades")
            for lab, p in probs.items():
                st.progress(float(p), text=f"{lab.capitalize()}: {float(p)*100:.2f}%")

        if st.button("🔄 Subir otra imagen", key="reset_img"):
            os.remove(temp_img)
            st.session_state.last_img_hash = None
            st.rerun()

# ========= TAB AUDIO =========
with tab_audio:
    st.markdown("### Sube un archivo de audio (WAV/MP3/FLAC) para detectar la emoción.")
    audio_file = st.file_uploader("🔊 Audio", type=["wav", "mp3", "flac"], key="audio_uploader")

    temp_audio = "temp_audio.wav"

    if audio_file:
        audio_bytes = audio_file.read()
        audio_hash  = md5(audio_bytes)

        if st.session_state.last_audio_hash != audio_hash:
            st.session_state.last_audio_hash = audio_hash

            ts = int(time.time())
            ext = os.path.splitext(audio_file.name)[1]
            saved_path = os.path.join(AUDIO_FOLDER, f"aud_{ts}{ext}")

            with open(saved_path, "wb") as f: f.write(audio_bytes)
            with open(temp_audio, "wb") as f: f.write(audio_bytes)

            st.rerun()

    if os.path.exists(temp_audio):
        st.audio(open(temp_audio, "rb").read(), format="audio/wav")

        with st.spinner("Analizando emoción..."):
            emo_a, probs_a = predict_emotion_audio(temp_audio)

        st.success(f"🎯 Emoción detectada: **{emo_a.upper()}**")

        if probs_a:
            st.markdown("#### 📊 Probabilidades")
            for lab, p in probs_a.items():
                st.progress(float(p), text=f"{lab.capitalize()}: {float(p)*100:.2f}%")

        if st.button("🔄 Subir otro audio", key="reset_audio"):
            os.remove(temp_audio)
            st.session_state.last_audio_hash = None
            st.rerun()
