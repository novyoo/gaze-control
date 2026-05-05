"""
run_phase2.py
Live webcam test for Phase 2.
"""

# ===================== IMPORTS =====================
import cv2
import time
import yaml
import numpy as np
from src.gaze.detector import FaceDetector
from src.gaze.face_data import FaceData
from src.gaze.estimator import GazeEstimator


# ===================== CONFIG LOADING =====================
def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


# ===================== DEBUG VISUALS =====================
def draw_debug(frame, face, gaze_xy, fps, estimator=None):
    h_frame, w_frame = frame.shape[:2]
    gx, gy = gaze_xy

    # ---------- Iris dots ----------
    cv2.circle(frame, tuple(face.left_iris.astype(int)),  5, (0, 255, 0), -1)
    cv2.circle(frame, tuple(face.right_iris.astype(int)), 5, (0, 255, 0), -1)

    # ---------- Eye corners ----------
    for corner in [*face.left_eye_corners, *face.right_eye_corners]:
        cv2.circle(frame, tuple(corner.astype(int)), 2, (255, 100, 0), -1)

    # ---------- Minimap ----------
    map_x, map_y, map_w, map_h = 10, 10, 180, 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (map_x, map_y),
                  (map_x + map_w, map_y + map_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (map_x, map_y),
                  (map_x + map_w, map_y + map_h), (80, 80, 80), 1)

    dot_x = int(map_x + gx * map_w)
    dot_y = int(map_y + gy * map_h)
    cv2.circle(frame, (dot_x, dot_y), 6, (140, 238, 255), -1)

    # ---------- HUD ----------
    blink_str = "YES" if (estimator and estimator.is_blinking) else "no"
    ear_val   = estimator._eye_aspect_ratio(face) if estimator else 0.0

    lines = [
        (f"FPS:        {fps:.1f}",           (255, 255,   0)),
        (f"Gaze X:     {gx:.3f}",            (255, 255, 255)),
        (f"Gaze Y:     {gy:.3f}",            (255, 255, 255)),
        (f"Confidence: {face.confidence:.2f}",(255, 255, 255)),
        (f"Blink:      {blink_str}",         (0, 180, 255) if blink_str == "YES" else (160,160,160)),
        (f"EAR:        {ear_val:.3f}",       (200, 200, 200)),
        ("Press N = set neutral",             (140, 238, 255)),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text,
                    (w_frame - 280, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    return frame


# ===================== MAIN LOOP =====================
def main():
    config    = load_config()
    detector  = FaceDetector(config)
    estimator = GazeEstimator()

    cap = cv2.VideoCapture(config["camera"]["index"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS,          config["camera"]["fps"])

    print("Phase 2 running")
    print("N = set neutral pose (look straight ahead first)")
    print("Q = quit")

    prev_time   = time.time()
    smooth_gaze = None
    fps         = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        face  = detector.process(frame)

        now         = time.time()
        instant_fps = 1.0 / max(now - prev_time, 1e-6)
        fps         = 0.9 * fps + 0.1 * instant_fps if fps != 0 else instant_fps
        prev_time   = now

        if face:
            new_gaze = estimator.estimate(face)

            if smooth_gaze is None:
                smooth_gaze = new_gaze

            alpha       = 0.2
            smooth_gaze = (
                smooth_gaze[0] * (1 - alpha) + new_gaze[0] * alpha,
                smooth_gaze[1] * (1 - alpha) + new_gaze[1] * alpha,
            )

            frame = draw_debug(frame, face, smooth_gaze, fps, estimator)

        else:
            cv2.putText(frame, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Gaze Control — Phase 2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            if face:
                estimator.calibrate_neutral(face)
                print("Neutral pose set! Now move your head and watch Gaze X/Y change.")
            else:
                print("No face detected — look at camera first")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()