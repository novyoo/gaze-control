# run_phase8.py
# Phase 8: 9-point calibration — now with head compensation
# Press N to set neutral | R to restart | Q to quit

import cv2
import time
import yaml
import numpy as np
import ctypes
from src.gaze.detector          import FaceDetector
from src.gaze.estimator         import GazeEstimator
from src.smoothing.kalman       import GazeKalmanFilter
from src.calibration.collector  import GazeSampleCollector
from src.calibration.calibrator import GazeCalibrator


def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def get_dot_pixel(point, sw, sh):
    padding_x = int(sw * 0.02)
    padding_y = int(sh * 0.02)
    px = int(point.screen_x * (sw - 2 * padding_x) + padding_x)
    py = int(point.screen_y * (sh - 2 * padding_y) + padding_y)
    return px, py


def draw_calibration(overlay, collector, state,
                     iris_px, sw, sh, fps,
                     head_stable, neutral_set):

    overlay[:] = 10

    # Completed dots
    for point in collector.completed_points:
        px, py = get_dot_pixel(point, sw, sh)
        cv2.circle(overlay, (px, py), 16, (0, 200, 80), -1)
        cv2.circle(overlay, (px, py),  6, (0, 255, 100), -1)

    # Remaining dots dim
    for i, point in enumerate(collector.points):
        if point.complete or i == collector.current:
            continue
        px, py = get_dot_pixel(point, sw, sh)
        cv2.circle(overlay, (px, py), 14, (50, 50, 50), 1)
        cv2.circle(overlay, (px, py),  4, (50, 50, 50), -1)

    # Current active dot
    if not collector.is_done:
        point    = collector.current_point
        px, py   = get_dot_pixel(point, sw, sh)
        progress = state.get("progress", 0.0)
        status   = state.get("status", "")

        # Pulsing ring
        pulse = int(20 + 8 * np.sin(time.time() * 4))
        cv2.circle(overlay, (px, py), pulse, (255, 255, 255), 1)

        # Progress arc
        if status == "collecting" and progress > 0:
            cv2.ellipse(overlay, (px, py), (18, 18),
                        -90, 0, int(360 * progress),
                        (0, 255, 100), 3)

        # Center dot color by status
        color = {
            "collecting":  (0,   255, 100),
            "hold_steady": (255, 255,   0),
            "stabilizing": (255, 180,   0),
            "get_ready":   (200, 200, 200),
            "blink":       (0,   100, 255),
        }.get(status, (255, 255, 255))

        cv2.circle(overlay, (px, py), 8, color, -1)

        idx = state.get("index", 0)
        cv2.putText(overlay,
                    f"{idx + 1}/{state.get('total', 9)}",
                    (px + 24, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1)

    # Iris position dot
    if iris_px:
        ix, iy = iris_px
        cv2.circle(overlay, (ix, iy), 5, (100, 100, 255), -1)

    # Status message
    if collector.is_done:
        cv2.putText(overlay, "Calibration complete!",
                    (sw//2 - 200, sh//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 100), 2)
        cv2.putText(overlay, "Saving... press Q to finish",
                    (sw//2 - 200, sh//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1)
    else:
        status = state.get("status", "")
        msgs = {
            "get_ready":   "Get ready — look at the dot when it appears",
            "blink":       "Keep eyes OPEN — stare at the dot",
            "stabilizing": "Hold your head STILL and stare at the dot",
            "hold_steady": "Good — keep staring...",
            "collecting":  "Collecting — keep staring!",
            "next":        "Point done! Find the next dot",
        }
        msg   = msgs.get(status, "Look at the dot")
        color = (0, 255, 100) if status == "collecting" else (200, 200, 200)
        cv2.putText(overlay, msg,
                    (sw//2 - 280, sh - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1)

    # Head stability indicator
    stab_color = (0, 255, 100) if head_stable else (0, 100, 255)
    stab_text  = "Head: STABLE" if head_stable else "Head: MOVING — hold still"
    cv2.putText(overlay, stab_text,
                (sw//2 - 160, sh - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, stab_color, 1)

    # Neutral indicator
    neutral_color = (0, 255, 100) if neutral_set else (255, 180, 0)
    neutral_text  = "Neutral: SET" if neutral_set else "Press N = set neutral (look straight ahead first)"
    cv2.putText(overlay, neutral_text,
                (sw//2 - 280, sh - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, neutral_color, 1)

    # HUD
    cv2.putText(overlay,
                f"FPS: {fps:.1f}  |  Points: {len(collector.completed_points)}/9  |  N=neutral  R=restart  Q=quit",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)


def main():
    config     = load_config()
    detector   = FaceDetector(config)
    estimator  = GazeEstimator()
    collector  = GazeSampleCollector(config)
    calibrator = GazeCalibrator(config)

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)

    cap = cv2.VideoCapture(config["camera"]["index"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS,          config["camera"]["fps"])

    overlay     = np.zeros((sh, sw, 3), dtype=np.uint8)
    prev_time   = time.time()
    fps         = 0.0
    state       = {}
    iris_px     = None
    saved       = False
    neutral_set = False

    print("Phase 8 — Calibration")
    print("1. Press N while looking straight at screen CENTER")
    print("2. Then stare at each dot until it fills green")
    print("3. Keep your head still — only move your EYES")
    print("R = restart | Q = quit")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        face  = detector.process(frame)

        now       = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        head_stable = False

        if face:
            # Get raw iris position (not compensated)
            raw_iris_x, raw_iris_y = estimator._raw_iris_position(face)
            head_yaw, head_pitch   = estimator._head_angles(face)
            is_blinking            = estimator.is_blinking

            # Run estimate to update head history
            estimator.estimate(face)
            head_stable = estimator.is_head_stable()

            # Show iris dot
            iris_px = (int(raw_iris_x * sw), int(raw_iris_y * sh))

            # Collect samples
            if not collector.is_done and neutral_set:
                state = collector.update(
                    raw_iris_x, raw_iris_y,
                    head_yaw, head_pitch,
                    is_blinking,
                    head_stable
                )

            # Auto save when done
            if collector.is_done and not saved:
                if calibrator.fit(collector):
                    calibrator.save()
                    saved = True
                    print("Calibration saved!")

        draw_calibration(overlay, collector, state,
                         iris_px, sw, sh, fps,
                         head_stable, neutral_set)

        cv2.imshow("Calibration", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            if face:
                estimator.set_neutral(face)
                neutral_set = True
                print("Neutral set — now stare at each dot")
            else:
                print("No face — look at camera first")
        elif key == ord('r'):
            collector   = GazeSampleCollector(config)
            calibrator  = GazeCalibrator(config)
            saved       = False
            neutral_set = False
            print("Restarted")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()