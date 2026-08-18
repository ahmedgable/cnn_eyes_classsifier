import io
import os
import tempfile
import time
import zipfile

try:
    import cv2
except ImportError:
    subprocess.run(["pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"])
    subprocess.run(["pip", "install", "opencv-python-headless"])
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
IMG_SIZE = (64, 64)
DEFAULT_MODEL_PATH = "saved_models/best_eye_model.keras"

# إعداد MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# نقاط العين اليسرى واليمنى وفقاً لمعايير MediaPipe Face Mesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def build_eye_state_classifier(input_shape=(64, 64, 1)):
    from keras import layers, models

    inputs = layers.Input(shape=input_shape, name="eye_input")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    x = layers.RandomRotation(factor=0.08)(x)
    x = layers.RandomZoom(height_factor=0.08)(x)

    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="eye_output")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="EyeStateClassifier")


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    if model_path.endswith(".tflite"):
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return ("tflite", interpreter)

    try:
        import keras
    except ImportError:
        import tensorflow as tf
        keras = tf.keras

    try:
        model = keras.models.load_model(model_path)
        return ("keras", model)
    except Exception:
        pass

    model = build_eye_state_classifier()
    with zipfile.ZipFile(model_path, "r") as zf:
        weights_entry = next((n for n in zf.namelist() if n.endswith(".weights.h5")), None)
        if weights_entry is None:
            raise RuntimeError("Could not find a weights file inside the .keras archive.")
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_path = zf.extract(weights_entry, tmpdir)
            model.load_weights(extracted_path)
    return ("keras", model)


def predict_eye(kind, model, eye_img_gray_64x64: np.ndarray) -> float:
    x = eye_img_gray_64x64.astype(np.float32).reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    if kind == "tflite":
        interpreter = model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], x.astype(input_details[0]["dtype"]))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        return float(out.reshape(-1)[0])
    else:
        out = model.predict(x, verbose=0)
        return float(out.reshape(-1)[0])


def crop_eye_region(frame, landmarks, eye_indices, padding=10):
    h, w, _ = frame.shape
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]

    xmin, xmax = max(0, min(x_coords) - padding), min(w, max(x_coords) + padding)
    ymin, ymax = max(0, min(y_coords) - padding), min(h, max(y_coords) + padding)

    if xmax <= xmin or ymax <= ymin:
        return None, None

    eye_crop = frame[ymin:ymax, xmin:xmax]
    return eye_crop, (xmin, ymin, xmax - xmin, ymax - ymin)


def annotate_frame_mediapipe(frame_bgr, kind, model, face_mesh, open_thresh: float):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    any_eye = False
    any_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            for eye_indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
                eye_crop, bbox = crop_eye_region(frame_bgr, landmarks, eye_indices)
                if eye_crop is None or eye_crop.size == 0:
                    continue

                any_eye = True
                gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
                resized_eye = cv2.resize(gray_eye, IMG_SIZE)

                prob_open = predict_eye(kind, model, resized_eye)
                is_open = prob_open > open_thresh

                if not is_open:
                    any_closed = True

                label = "Open" if is_open else "Closed"
                color = (0, 200, 0) if is_open else (0, 0, 255)

                x, y, w_box, h_box = bbox
                cv2.rectangle(frame_bgr, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(frame_bgr, f"{label} ({prob_open:.2f})", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return frame_bgr, any_eye, any_closed


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Eye State Detector", page_icon="👁️", layout="centered")
st.title("👁️ Eye State / Drowsiness Detector")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    open_thresh = st.slider("Open probability threshold", 0.0, 1.0, 0.5, 0.05)
    closed_seconds_alert = st.slider("Drowsiness alert after (seconds closed)", 0.5, 5.0, 1.5, 0.5)

if not os.path.exists(model_path):
    st.error(f"Model file not found: `{model_path}`. Update the path in the sidebar.")
    st.stop()

kind, model = load_model(model_path)
st.success(f"Model loaded ({kind}) ✅")

tab_live, tab_upload = st.tabs(["📷 Live Webcam", "🖼️ Upload Image"])

# ------------------------------------------------------------------
# TAB 1: Live webcam feed
# ------------------------------------------------------------------
with tab_live:
    st.write("Click **Start** to open your webcam and run real-time detection.")
    run = st.toggle("Start webcam", value=False, key="run_webcam")
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam (index 0). Is it connected / not in use by another app?")
        else:
            closed_since = None
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                try:
                    while st.session_state.get("run_webcam", False):
                        ok, frame = cap.read()
                        if not ok:
                            st.warning("Failed to read frame from webcam.")
                            break

                        annotated, any_eye, any_closed = annotate_frame_mediapipe(
                            frame, kind, model, face_mesh, open_thresh
                        )

                        if any_eye and any_closed:
                            if closed_since is None:
                                closed_since = time.time()
                            elapsed = time.time() - closed_since
                            if elapsed >= closed_seconds_alert:
                                alert_placeholder.error("🚨 DROWSINESS ALERT!")
                            else:
                                alert_placeholder.empty()
                        else:
                            closed_since = None
                            alert_placeholder.empty()

                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                finally:
                    cap.release()
    else:
        frame_placeholder.info("Webcam is stopped. Toggle 'Start webcam' above to begin.")

# ------------------------------------------------------------------
# TAB 2: Upload image
# ------------------------------------------------------------------
with tab_upload:
    st.write("Upload a photo (a face or a close-up of an eye) to check its state.")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            annotated, any_eye, any_closed = annotate_frame_mediapipe(
                frame_bgr.copy(), kind, model, face_mesh, open_thresh
            )

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection result",
                  use_container_width=True)

        if not any_eye:
            st.warning("No eyes were detected in this image. Try a clearer, more frontal photo.")
        elif any_closed:
            st.error("At least one detected eye is Closed.")
        else:
            st.success("All detected eyes are Open.")
