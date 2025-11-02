# main.py — Real-time Emotion Detection (clean Streamlit version)

import cv2 # type: ignore
import numpy as np # type: ignore
import streamlit as st # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# --------------------------- CONFIG ---------------------------
MODEL_PATH = r"data\model\emotion_model.h5"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EMOTIONS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --------------------------- LOAD MODEL ---------------------------
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# --------------------------- PAGE SETUP ---------------------------
st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="centered")

st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: white; text-align: center; }
    h1 { color: #ffcc00; text-align: center; }
    .note { color: #aaa; font-size: 0.9rem; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎭 Real-Time Facial Emotion Detection")
st.markdown("<div class='note'>Powered by CNN</div>", unsafe_allow_html=True)

# --------------------------- WEBCAM STREAM ---------------------------
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image(np.zeros((240, 320, 3), dtype=np.uint8), caption="Live Feed", use_container_width=True)

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        label = EMOTIONS[np.argmax(preds)]
        confidence = np.max(preds) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

cap.release()

