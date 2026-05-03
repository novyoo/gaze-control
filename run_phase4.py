# run_phase4.py
# Phase 4: real cursor control from gaze
# IMPORTANT: move mouse to top-left corner to emergency stop (PyAutoGUI failsafe)
# Press Q in the overlay window to quit cleanly

import cv2
import time
import yaml
import numpy as np
from src.gaze.detector   import FaceDetector
from src.gaze.estimator  import GazeEstimator
from src.cursor.mapper     import GazeMapper
from src.cursor.controller import CursorController


def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def draw_hud(overlay, face_found, raw_gaze, smooth_pos,
             fps, is_blinking, ear, controller):

    sw = overlay.shape[1]
    sh = overlay.shape[0]

    # Grid
    for gx in [sw//4, sw//2, 3*sw//4]:
        cv2.line(overlay, (gx, 0), (gx, sh), (35, 35, 35), 1)
    for gy in [sh//4, sh//2, 3*sh//4]:
        cv2.line(overlay, (0, gy), (sw, gy), (35, 35, 35), 1)

    if face_found and not is_blinking:
        sx, sy = smooth_pos

        # Outer ring + center dot
        cv2.circle(overlay, (sx, sy), 28, (0, 200, 0), 1)
        cv2.circle(overlay, (sx, sy), 6,  (0, 255, 0), -1)

        # Crosshair lines
        cv2.line(overlay, (sx - 45, sy), (sx + 45, sy), (0, 255, 0), 1)
        cv2.line(overlay, (sx, sy - 45), (sx, sy + 45), (0, 255, 0), 1)

    elif is_blinking:
        # Show frozen indicator when blinking
        sx, sy = smooth_pos
        cv2.circle(overlay, (sx, sy), 28, (0, 180, 255), 1)
        cv2.circle(overlay, (sx, sy), 6,  (0, 180, 255), -1)

    # --- Left HUD ---
    status = "ON" if controller.is_enabled else "OFF"
    blink_str = "YES" if is_blinking else "no"

    left_lines = [
        ("CURSOR CTRL", (160, 160, 160)),
        (f"status: {status}",  (0, 255, 100) if controller.is_enabled else (0, 80, 255)),
        ("", None),
        ("RAW GAZE", (160, 160, 160)),
        (f"gx: {raw_gaze[0]:.4f}", (255, 255, 255)),
        (f"gy: {raw_gaze[1]:.4f}", (255, 255, 255)),
        ("", None),
        ("SMOOTH POS", (160, 160, 160)),
        (f"px: {smooth_pos[0]}", (0, 255, 200)),
        (f"py: {smooth_pos[1]}", (0, 255, 200)),
        ("", None),
        (f"EAR:   {ear:.3f}",       (255, 255, 255)),
        (f"Blink: {blink_str}", (0, 180, 255) if is_blinking else (255, 255, 255)),
    ]

    y = 30
    for text, color in left_lines:
        if text and color:
            cv2.putText(overlay, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        y += 22

    # --- Top right: FPS ---
    cv2.putText(overlay, f"FPS: {fps:.1f}", (sw - 160, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

    # --- Bottom hint ---
    cv2.putText(overlay,
                "T = toggle cursor | Q = quit | move mouse to top-left = emergency stop",
                (20, sh - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)


def main():
    config     = load_config()
    detector   = FaceDetector(config)
    estimator  = GazeEstimator()
    mapper     = GazeMapper(config)
    controller = CursorController(config)

    sw, sh = mapper.screen_w, mapper.screen_h

    cap = cv2.VideoCapture(config["camera"]["index"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS,          config["camera"]["fps"])

    overlay   = np.zeros((sh, sw, 3), dtype=np.uint8)
    prev_time = time.time()
    fps       = 0.0
    raw_gaze  = (0.5, 0.5)
    smooth_pos = (sw // 2, sh // 2)

    print("Phase 4 running")
    print("T = toggle cursor on/off")
    print("Move mouse to top-left corner = emergency stop")
    print("Q = quit")

    cv2.namedWindow("Gaze Control", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Control",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        face  = detector.process(frame)

        # FPS
        now   = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        overlay[:] = 18   # near-black background

        is_blinking = False
        ear = 0.0

        if face:
            raw_gaze    = estimator.estimate(face)
            is_blinking = estimator.is_blinking
            ear         = estimator.ear_value(face)

            # Map to screen
            target_x, target_y = mapper.map(*raw_gaze)

            # Move cursor (skipped during blink)
            if not is_blinking:
                smooth_pos = controller.move(target_x, target_y)
            else:
                smooth_pos = controller.get_smooth_pos()

        draw_hud(overlay, face is not None,
                 raw_gaze, smooth_pos,
                 fps, is_blinking, ear, controller)

        # Camera preview bottom-right
        prev_w, prev_h = 240, 135
        small = cv2.resize(frame, (prev_w, prev_h))
        overlay[sh - prev_h - 10: sh - 10,
                sw - prev_w - 10: sw - 10] = small

        cv2.imshow("Gaze Control", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            if controller.is_enabled:
                controller.disable()
                print("Cursor control DISABLED")
            else:
                controller.enable()
                print("Cursor control ENABLED")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()