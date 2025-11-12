"""
Streamlit app to control screen brightness using hand gestures.

Requirements:
pip install streamlit opencv-python mediapipe numpy screen_brightness_control

Notes:
- On Streamlit Cloud the standard webcam access may be blocked; Streamlit Cloud does not
  provide direct webcam access in all cases. If the camera doesn't open on the Cloud,
  run this locally (streamlit run app.py) to test.
- If you prefer to use frames captured via st.camera_input (single photos), we can adapt.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Attempt to import screen_brightness_control (cross-platform)
try:
    import screen_brightness_control as sbc
    SB_AVAILABLE = True
except Exception as e:
    SB_AVAILABLE = False

st.set_page_config(page_title="Gesture Brightness Control", layout="centered")

st.title("🔆 Gesture-based Brightness Control")
st.markdown(
    """
Move your thumb and index finger to change screen brightness.
- Hold index finger and thumb in view.
- Increase distance → brightness up. Decrease distance → brightness down.
"""
)

# Sidebar controls
st.sidebar.header("Settings")
run_camera = st.sidebar.checkbox("Run camera", value=False)
sensitivity = st.sidebar.slider("Sensitivity (mapping smoothing)", 1, 10, 4)
smoothing = st.sidebar.slider("Smoothing (reduce jitter)", 1, 20, 5)
min_dist = st.sidebar.slider("Min gesture distance (px)", 5, 30, 10)
max_dist = st.sidebar.slider("Max gesture distance (px)", 100, 400, 200)
allow_system_change = st.sidebar.checkbox("Allow system brightness change", value=SB_AVAILABLE)

if not SB_AVAILABLE:
    st.sidebar.warning(
        "screen_brightness_control not available. Install with `pip install screen_brightness_control`.\n"
        "App will show computed brightness but won't change system brightness."
    )

# Video display area
frame_slot = st.empty()
info_slot = st.empty()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# helper functions
def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def map_value(x, in_min, in_max, out_min, out_max):
    # clamp
    if x < in_min: x = in_min
    if x > in_max: x = in_max
    return (x - in_min) * (out_max - out_min) / (in_max - in_min + 1e-9) + out_min

# smoothing state
prev_brightness = None
prev_time = 0.0

if run_camera:
    stframe = frame_slot
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Try running locally with camera access.")
    else:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            st.success("Camera started. Show thumb and index finger.")
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from camera.")
                    break

                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                brightness_percent = None
                draw_info = "No hand detected"

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    # get landmark pixel coordinates
                    lm = []
                    for id, lm_pt in enumerate(hand.landmark):
                        lm.append((int(lm_pt.x * w), int(lm_pt.y * h)))

                    # index finger tip is id 8, thumb tip is id 4
                    index_tip = lm[8]
                    thumb_tip = lm[4]

                    # draw landmarks
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    # draw circle and line
                    cv2.circle(frame, index_tip, 8, (255, 0, 255), -1)
                    cv2.circle(frame, thumb_tip, 8, (255, 0, 255), -1)
                    cv2.line(frame, index_tip, thumb_tip, (255, 0, 255), 2)

                    # compute distance
                    dist = euclidean(index_tip, thumb_tip)

                    # map distance to brightness 0-100
                    # use user min_dist and max_dist
                    mapped = map_value(dist, min_dist, max_dist, 0, 100)
                    # smoothing
                    if prev_brightness is None:
                        smoothed = mapped
                    else:
                        alpha = 1.0 / max(1, smoothing)  # higher smoothing -> smaller alpha
                        smoothed = prev_brightness * (1 - alpha) + mapped * alpha

                    brightness_percent = int(np.clip(smoothed, 0, 100))
                    prev_brightness = smoothed
                    draw_info = f"Dist: {int(dist)} px → Brightness: {brightness_percent}%"

                    # set system brightness if allowed
                    if allow_system_change and SB_AVAILABLE:
                        try:
                            # apply with small step/sensitivity to avoid thrashing
                            # we apply only if absolute change above threshold derived from sensitivity
                            apply_threshold = sensitivity
                            current_system = None
                            try:
                                current_system = sbc.get_brightness(display=0)
                                if isinstance(current_system, list):
                                    if len(current_system)>0:
                                        current_system = current_system[0]
                                if current_system is None:
                                    current_system = 0
                                # Only set when change is meaningful
                                if abs(int(current_system) - brightness_percent) >= apply_threshold:
                                    sbc.set_brightness(brightness_percent)
                            except Exception:
                                # fallback: attempt set without read
                                sbc.set_brightness(brightness_percent)
                        except Exception as e:
                            # Do not crash app for brightness API errors
                            draw_info += f" (system change failed: {e})"

                # show info and frame
                info_slot.markdown(f"**{draw_info}**")
                # convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

                # small sleep to reduce CPU
                time.sleep(0.02)

                # update run_camera in case user toggles sidebar (streamlit can't check directly - we break on next re-run)
                run_camera = st.sidebar.checkbox("Run camera", value=True)

        cap.release()
        st.success("Camera stopped.")
else:
    st.info("Toggle 'Run camera' in the sidebar to start webcam and control brightness.")
