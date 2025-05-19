import streamlit as st
import os
import sys
import time
from PIL import Image
import hashlib

# Añadir ruta del directorio raíz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from src.detect_emotion_image import predict_emotion_image

# Configuración de la página
st.set_page_config(page_title="Detector de Emociones", layout="centered")
st.title("😄 Reconocimiento de Emociones Faciales")
st.markdown("Sube una imagen con un rostro visible para detectar la emoción expresada.")

# Crear carpeta de imágenes
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
temp_path = "temp.jpg"

# Para evitar problemas de caché, usamos un hash
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# Estado de sesión
if 'last_image_hash' not in st.session_state:
    st.session_state.last_image_hash = None

# Subida de imagen
uploaded_file = st.file_uploader("📷 Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    current_hash = get_file_hash(file_bytes)

    if st.session_state.last_image_hash != current_hash:
        st.session_state.last_image_hash = current_hash

        timestamp = int(time.time())
        file_ext = os.path.splitext(uploaded_file.name)[1]
        saved_image_path = os.path.join(UPLOAD_FOLDER, f"imagen_{timestamp}{file_ext}")

        # Guardar la imagen
        with open(saved_image_path, "wb") as f:
            f.write(file_bytes)
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        st.rerun()

# Procesar si hay una imagen temporal
if os.path.exists(temp_path):
    st.image(Image.open(temp_path), caption="Imagen subida", use_container_width=True)

    with st.spinner("Detectando emoción..."):
        emotion, processed_img, probs = predict_emotion_image(temp_path)

    st.success(f"🎯 Emoción detectada: **{emotion.upper()}**")

    # Mostrar imagen procesada
    if processed_img is not None:
        import matplotlib.pyplot as plt
        st.markdown("### 🧠 Imagen procesada")
        fig, ax = plt.subplots()
        ax.imshow(processed_img)
        ax.axis("off")
        st.pyplot(fig)

    # Mostrar barras de probabilidad
    if probs is not None:
        st.markdown("### 📊 Probabilidades por emoción")
        for label, prob in probs.items():
            st.progress(float(prob) / 100.0, text=f"{label.capitalize()}: {float(prob):.2f}%")


