# src/predict_emotion_audio.py
import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoProcessor

# Modelo SUPERB ER (Emotion Recognition)
MODEL_ID = "superb/wav2vec2-base-superb-er"

# Carga pesos y procesador (se descargan la 1.ª vez)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

# Etiquetas que maneja el modelo
EMO_LABELS = [
    "angry", "contempt", "disgusted",
    "fearful", "happy", "neutral",
    "sad", "surprised"
]

def predict_emotion_audio(wav_path):
    # ── 1. Cargar audio ───────────────────────────────
    signal, sr = torchaudio.load(wav_path)

    # Mono & 16 kHz para el modelo
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    if signal.shape[0] > 1:          # estéreo → mono
        signal = signal.mean(dim=0, keepdim=True)

    # ── 2. Preparar entrada para Transformers ─────────
    inputs = processor(signal.squeeze(), sampling_rate=16000, return_tensors="pt")

    # ── 3. Inferencia ─────────────────────────────────
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0)

    # ── 4. Formatear salida ───────────────────────────
    probs_dict = {lab: float(probs[i]) for i, lab in enumerate(EMO_LABELS)}
    dominant = max(probs_dict, key=probs_dict.get)

    return dominant, probs_dict
