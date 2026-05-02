"""
run_phase2.py
Live webcam test for Phase 2.
"""

# ===================== IMPORTS =====================
import cv2
import time
import yaml
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from src.gaze.detector import FaceDetector
from src.gaze.estimator import GazeEstimator


# ===================== CONFIG LOADING =====================
def load_config():
    """Load YAML configuration file"""
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


# ===================== PNG OVERLAY =====================
def overlay_png(frame, png, x, y):
    """Overlay transparent PNG onto frame"""
    h, w = png.shape[:2]

    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame

    overlay = png[:, :, :3]
    mask = png[:, :, 3:] / 255.0

    roi = frame[y:y+h, x:x+w]
    frame[y:y+h, x:x+w] = (1 - mask) * roi + mask * overlay

    return frame


# ===================== EYE GLOW =====================
def draw_glow(frame, center, color):
    """Subtle glow around eye landmarks"""
    x, y = center

    for r in range(4, 0, -2):
        alpha = 0.05
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), r * 2, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.circle(frame, (x, y), 1, color, -1)


# ===================== CUSTOM FONT TEXT =====================
def draw_text_pil(frame, text, pos,
                  font_path="assets/Daydreaming-Bold.otf",
                  size=20,
                  color=(140, 238, 255)):  # #FFEE8C in BGR
    """Draw text using custom font via PIL"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, size)

    # PIL uses RGB → convert BGR to RGB
    draw.text(pos, text, font=font, fill=color[::-1])

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ===================== DEBUG VISUALS =====================
def draw_debug(frame, face, gaze_xy, fps, gaze_icon):
    """Render all visual overlays (eyes, minimap, HUD)"""
    h_frame, w_frame = frame.shape[:2]
    gx, gy = gaze_xy

    # ---------- Iris Icons ----------
    h, w = gaze_icon.shape[:2]

    x, y = face.left_iris.astype(int)
    frame = overlay_png(frame, gaze_icon, x - w // 2, y - h // 2)

    x, y = face.right_iris.astype(int)
    frame = overlay_png(frame, gaze_icon, x - w // 2, y - h // 2)

    # ---------- Eye Glow ----------
    for corner in [*face.left_eye_corners, *face.right_eye_corners]:
        draw_glow(frame, tuple(corner.astype(int)), (140, 238, 255))

    # ---------- Minimap ----------
    map_x, map_y, map_w, map_h = 10, 10, 180, 110

    overlay = frame.copy()
    cv2.rectangle(overlay, (map_x, map_y), (map_x + map_w, map_y + map_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.rectangle(frame, (map_x, map_y), (map_x + map_w, map_y + map_h), (80, 80, 80), 1)

    dot_x = int(map_x + gx * map_w)
    dot_y = int(map_y + gy * map_h)

    cv2.circle(frame, (dot_x, dot_y), 6, (140, 238, 255), -1)

    # ---------- HUD TEXT ----------
    lines = [
        f"FPS: {fps:.1f}",
        f"Gaze X: {gx:.3f}",
        f"Gaze Y: {gy:.3f}",
        f"Confidence: {face.confidence:.2f}",
    ]

    for i, line in enumerate(lines):
        x = w_frame - 220
        y = 30 + i * 28

        frame = draw_text_pil(
            frame,
            line,
            (x, y),
            font_path="assets/Daydreaming-Bold.otf",
            size=20,
            color=(140, 238, 255)  # Soft yellow (#FFEE8C)
        )

    return frame


# ===================== MAIN LOOP =====================
def main():
    config = load_config()

    detector = FaceDetector(config)
    estimator = GazeEstimator()

    # ---------- Load Assets ----------
    gaze_icon = cv2.imread("assets/gaze_dot.png", cv2.IMREAD_UNCHANGED)
    gaze_icon = cv2.resize(gaze_icon, (16, 16))

    # ---------- Camera Setup ----------
    cap = cv2.VideoCapture(config["camera"]["index"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, config["camera"]["fps"])

    print("Phase 2 running — press Q to quit")

    prev_time = time.time()
    smooth_gaze = None
    fps = 0

    # ---------- Main Loop ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        face = detector.process(frame)

        # ---------- FPS Calculation ----------
        now = time.time()
        instant_fps = 1.0 / max(now - prev_time, 1e-6)
        fps = 0.9 * fps + 0.1 * instant_fps if fps != 0 else instant_fps
        prev_time = now

        # ---------- Gaze Processing ----------
        if face:
            new_gaze = estimator.estimate(face)

            if smooth_gaze is None:
                smooth_gaze = new_gaze

            alpha = 0.2  # smoothing factor
            smooth_gaze = (
                smooth_gaze[0] * (1 - alpha) + new_gaze[0] * alpha,
                smooth_gaze[1] * (1 - alpha) + new_gaze[1] * alpha,
            )

            gaze_xy = smooth_gaze

            frame = draw_debug(frame, face, gaze_xy, fps, gaze_icon)

        else:
            cv2.putText(frame, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ---------- Display ----------
        cv2.imshow("Gaze Control — Phase 2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- Cleanup ----------
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    main()