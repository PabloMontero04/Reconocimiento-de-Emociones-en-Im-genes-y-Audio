# src/predict_emotion_audio.py

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


# Cargar extractor de características y modelo
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

# Etiquetas del modelo
EMO_LABELS = list(model.config.id2label.values())

def predict_emotion_audio(wav_path):
    # Cargar el audio
    signal, sr = torchaudio.load(wav_path)

    # Convertir a mono y resamplear
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)

    # Extraer características
    inputs = feature_extractor(signal.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    # Inferencia
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0)

    # Procesar resultados
    probs_dict = {label: float(probs[i]) * 100 for i, label in enumerate(EMO_LABELS)}
    dominant = max(probs_dict, key=probs_dict.get)

    return dominant, probs_dict
