import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from math import hypot

# optional: brightness control (works only locally)
try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except ImportError:
    BRIGHTNESS_AVAILABLE = False

st.set_page_config(page_title="Gesture Brightness Control", layout="centered")

st.title("🔆 Gesture-Based Brightness Control")
st.write("Move your thumb and index finger apart or together to adjust brightness!")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=1,
)
mp_draw = mp.solutions.drawing_utils

# Capture image from user camera (Streamlit cloud-friendly)
img_file = st.camera_input("📷 Take a photo (show your hand with thumb and index finger visible)")

if img_file is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    result = hands.process(frame_rgb)
    landmark_list = []

    if result.multi_hand_landmarks:
        for handlm in result.multi_hand_landmarks:
            for _id, lm in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_list.append([_id, x, y])
            mp_draw.draw_landmarks(frame, handlm, mp_hands.HAND_CONNECTIONS)

    brightness = None
    if landmark_list != []:
        x1, y1 = landmark_list[4][1], landmark_list[4][2]   # Thumb tip
        x2, y2 = landmark_list[8][1], landmark_list[8][2]   # Index tip

        # Draw visual indicators
        cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate distance between thumb & index finger
        length = hypot(x2 - x1, y2 - y1)

        # Map range (15–220 px) → (0–100 brightness)
        brightness = np.interp(length, [15, 220], [0, 100])

        # Set brightness (if available)
        if BRIGHTNESS_AVAILABLE:
            sbc.set_brightness(int(brightness))

    # Display the processed image
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    if brightness is not None:
        st.success(f"Estimated Brightness Level: **{int(brightness)}%**")
    else:
        st.warning("No hand detected. Try retaking the photo!")

else:
    st.info("👆 Click the camera above to capture your hand and adjust brightness!")

st.markdown("---")
st.caption("Developed by Suprith K — Powered by OpenCV & Mediapipe ✨")
