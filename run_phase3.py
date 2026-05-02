"""
run_phase3.py
Phase 3 test: shows a fullscreen gaze dot overlay.
- Green dot = where your gaze maps to on screen
- Top-left: raw gaze values
- Top-right: mapped screen coords
Press Q to quit.
"""

import cv2
import time
import yaml
import numpy as np
from src.gaze.detector import FaceDetector
from src.gaze.estimator import GazeEstimator
from src.cursor.mapper import GazeMapper


def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    detector  = FaceDetector(config)
    estimator = GazeEstimator()
    mapper    = GazeMapper(config)

    sw, sh = mapper.screen_w, mapper.screen_h

    cap = cv2.VideoCapture(config["camera"]["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS,          config["camera"]["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Create overlay canvas (black, full screen size)
    overlay = np.zeros((sh, sw, 3), dtype=np.uint8)

    # Camera preview is scaled down and shown in bottom-right
    prev_w, prev_h = 320, 180

    prev_time = time.time()
    prev_px, prev_py = sw // 2, sh // 2   # start at center

    print(f"Phase 3 running on {sw}x{sh} — press Q to quit")
    cv2.namedWindow("Gaze Overlay", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        face  = detector.process(frame)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # --- Reset overlay ---
        overlay[:] = 20   # dark gray background

        # --- Grid lines (visual reference) ---
        for gx in [sw//4, sw//2, 3*sw//4]:
            cv2.line(overlay, (gx, 0), (gx, sh), (40, 40, 40), 1)
        for gy in [sh//4, sh//2, 3*sh//4]:
            cv2.line(overlay, (0, gy), (sw, gy), (40, 40, 40), 1)

        if face:
            raw_gx, raw_gy = estimator.estimate(face)
            px, py = mapper.map(raw_gx, raw_gy)

            # Smooth movement trail (5 prev positions)
            alpha = 0.25
            smooth_px = int(alpha * px + (1 - alpha) * prev_px)
            smooth_py = int(alpha * py + (1 - alpha) * prev_py)
            prev_px, prev_py = smooth_px, smooth_py

            # --- Draw gaze dot + crosshair ---
            cv2.circle(overlay, (smooth_px, smooth_py), 30, (0, 180, 0), 2)    # outer ring
            cv2.circle(overlay, (smooth_px, smooth_py), 6,  (0, 255, 0), -1)   # center dot
            cv2.line(overlay, (smooth_px - 50, smooth_py), (smooth_px + 50, smooth_py), (0, 255, 0), 1)
            cv2.line(overlay, (smooth_px, smooth_py - 50), (smooth_px, smooth_py + 50), (0, 255, 0), 1)

            # --- HUD: raw values (top-left) ---
            cv2.putText(overlay, "RAW GAZE",      (20, 36),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
            cv2.putText(overlay, f"gx: {raw_gx:.4f}", (20, 62),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(overlay, f"gy: {raw_gy:.4f}", (20, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # --- HUD: mapped values (top-left continued) ---
            cv2.putText(overlay, "MAPPED SCREEN", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
            cv2.putText(overlay, f"px: {smooth_px}", (20, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
            cv2.putText(overlay, f"py: {smooth_py}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)

        else:
            cv2.putText(overlay, "No face detected — look at camera",
                        (sw // 2 - 280, sh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)

        # --- FPS (top-right) ---
        cv2.putText(overlay, f"FPS: {fps:.1f}", (sw - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

        # --- Camera preview (bottom-right corner) ---
        small = cv2.resize(frame, (prev_w, prev_h))
        overlay[sh - prev_h - 10 : sh - 10,
                sw - prev_w - 10 : sw - 10] = small

        cv2.imshow("Gaze Overlay", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()