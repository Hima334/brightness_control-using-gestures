# app.py  (cv2-free version)
import streamlit as st
from PIL import Image
import numpy as np
import mediapipe as mp
from math import hypot

# optional: brightness control (works only locally)
try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except Exception:
    BRIGHTNESS_AVAILABLE = False

st.set_page_config(page_title="Gesture Brightness Control", layout="centered")
st.title("🔆 Gesture-Based Brightness Control (no cv2 required)")
st.write("Take a photo (show thumb & index finger).")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1,
)
mp_draw = mp.solutions.drawing_utils

img_file = st.camera_input("📷 Take a photo")

if img_file is not None:
    # Read image bytes into PIL then to RGB numpy array
    image = Image.open(img_file).convert("RGB")
    frame = np.array(image)  # shape (H, W, 3), RGB
    frame_rgb = frame.copy()

    # Mediapipe expects RGB numpy array
    results = hands.process(frame_rgb)

    landmark_list = []
    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            # Collect landmark pixel coordinates
            h, w, _ = frame.shape
            for _id, lm in enumerate(handlm.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_list.append([_id, x, y])
            # Draw landmarks (this modifies 'frame' visually using mediapipe's drawing utils)
            mp_draw.draw_landmarks(frame, handlm, mp_hands.HAND_CONNECTIONS)

    brightness = None
    if landmark_list:
        x1, y1 = landmark_list[4][1], landmark_list[4][2]   # thumb tip
        x2, y2 = landmark_list[8][1], landmark_list[8][2]   # index tip
        length = hypot(x2 - x1, y2 - y1)
        brightness = np.interp(length, [15, 220], [0, 100])

        if BRIGHTNESS_AVAILABLE:
            try:
                sbc.set_brightness(int(brightness))
            except Exception as e:
                st.warning(f"Could not set system brightness: {e}")

    # Show the result image (PIL expects RGB)
    st.image(frame, channels="RGB", caption="Processed image (landmarks)")

    if brightness is not None:
        st.success(f"Estimated Brightness Level: **{int(brightness)}%**")
    else:
        st.warning("No hand detected. Try again.")
else:
    st.info("Click the camera above to take a photo.")

st.caption("No-cv2 version — uses Pillow + mediapipe.")
