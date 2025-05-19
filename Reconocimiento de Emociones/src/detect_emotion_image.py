import cv2
from deepface import DeepFace

def predict_emotion_image(image_path):
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        return "Imagen no válida", None, None

    # Detectar emociones usando DeepFace
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True)
        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_probabilities = analysis[0]['emotion']
    except Exception as e:
        return f"Error al analizar: {str(e)}", None, None

    # Redimensionar imagen para mostrarla en la app
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_display, (256, 256), interpolation=cv2.INTER_CUBIC)

    return dominant_emotion, img_display, emotion_probabilities
