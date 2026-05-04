# run_phase8.py
# Phase 8: 9-point calibration
# Stare at each dot until it fills — system learns your gaze
# Press R to restart | Q to quit the screen

import cv2
import time
import yaml
import numpy as np
from src.gaze.detector        import FaceDetector
from src.gaze.estimator       import GazeEstimator
from src.smoothing.kalman     import GazeKalmanFilter
from src.calibration.collector  import GazeSampleCollector
from src.calibration.calibrator import GazeCalibrator


def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def get_dot_pixel(point, sw, sh):
    """
    Convert normalized calibration point to pixel coords.
    Padding scales with screen so dots are fully visible on any resolution.
    """
    padding_x = int(sw * 0.02)   # 2% of screen width
    padding_y = int(sh * 0.02)   # 2% of screen height

    px = int(point.screen_x * (sw - 2 * padding_x) + padding_x)
    py = int(point.screen_y * (sh - 2 * padding_y) + padding_y)
    return px, py


def draw_calibration(overlay, collector, state,
                     gaze_px, sw, sh, fps):
    overlay[:] = 10   # near black

    # ── Completed dots — green ────────────────────────────────────────
    for point in collector.completed_points:
        px, py = get_dot_pixel(point, sw, sh)
        cv2.circle(overlay, (px, py), 14, (0, 200, 80), -1)
        cv2.circle(overlay, (px, py),  6, (0, 255, 100), -1)

    # ── Remaining dots — dim ──────────────────────────────────────────
    for i, point in enumerate(collector.points):
        if point.complete:
            continue
        if i == collector.current:
            continue
        px, py = get_dot_pixel(point, sw, sh)
        cv2.circle(overlay, (px, py), 14, (60, 60, 60), 1)
        cv2.circle(overlay, (px, py),  4, (60, 60, 60), -1)

    # ── Current active dot ────────────────────────────────────────────
    if not collector.is_done:
        point  = collector.current_point
        px, py = get_dot_pixel(point, sw, sh)
        progress = state.get("progress", 0.0)
        status   = state.get("status", "waiting")

        # Outer pulsing ring
        pulse = int(20 + 8 * np.sin(time.time() * 4))
        cv2.circle(overlay, (px, py), pulse, (255, 255, 255), 1)

        # Progress arc — fills as you hold gaze
        if status == "collecting" and progress > 0:
            angle = int(360 * progress)
            axes  = (18, 18)
            cv2.ellipse(overlay, (px, py), axes, -90, 0, angle,
                        (0, 255, 100), 3)

        # Center dot
        color = (0, 255, 255) if status == "collecting" else (255, 255, 255)
        cv2.circle(overlay, (px, py), 8, color, -1)

        # Point number label
        idx = state.get("index", 0)
        cv2.putText(overlay,
                    f"{idx + 1}/{state.get('total', 9)}",
                    (px + 24, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # ── Gaze dot — where eyes are right now ───────────────────────────
    if gaze_px:
        gx, gy = gaze_px
        cv2.circle(overlay, (gx, gy), 5, (100, 100, 255), -1)

    # ── Instructions ──────────────────────────────────────────────────
    if collector.is_done:
        cv2.putText(overlay, "Calibration complete!",
                    (sw//2 - 180, sh//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2)
        cv2.putText(overlay, "Saving... press Q to finish",
                    (sw//2 - 200, sh//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    else:
        status = state.get("status", "")
        if status == "blink":
            msg = "Keep eyes open and stare at the dot"
        elif status == "waiting":
            msg = "Look at the white dot"
        elif status == "collecting":
            msg = "Hold your gaze steady..."
        else:
            msg = "Get ready for next point"

        cv2.putText(overlay, msg,
                    (sw//2 - 200, sh - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

    # ── HUD ───────────────────────────────────────────────────────────
    done_count = len(collector.completed_points)
    cv2.putText(overlay,
                f"FPS: {fps:.1f}  |  Points: {done_count}/9",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

    cv2.putText(overlay, "R = restart  |  Q = quit",
                (sw - 260, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)


def main():
    config     = load_config()
    detector   = FaceDetector(config)
    estimator  = GazeEstimator()
    kalman     = GazeKalmanFilter(config)
    collector  = GazeSampleCollector(config)
    calibrator = GazeCalibrator(config)

    import ctypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)

    cap = cv2.VideoCapture(config["camera"]["index"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS,          config["camera"]["fps"])

    overlay   = np.zeros((sh, sw, 3), dtype=np.uint8)
    prev_time = time.time()
    fps       = 0.0
    state     = {}
    gaze_px   = None
    saved     = False

    print("Phase 8 — Calibration")
    print("Stare at each dot until it fills green")
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

        if face:
            raw_gaze    = estimator.estimate(face)
            is_blinking = estimator.is_blinking

            # Kalman smooth the gaze
            sx, sy = kalman.update(raw_gaze[0], raw_gaze[1])

            # Show gaze dot position on screen
            gaze_px = (int(sx * sw), int(sy * sh))

            # Feed to collector
            if not collector.is_done:
                state = collector.update(sx, sy, is_blinking)

            # Auto-save when done
            if collector.is_done and not saved:
                if calibrator.fit(collector):
                    calibrator.save()
                    saved = True
                    print("Calibration saved!")

        draw_calibration(overlay, collector, state,
                         gaze_px, sw, sh, fps)

        cv2.imshow("Calibration", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Restart calibration
            collector  = GazeSampleCollector(config)
            calibrator = GazeCalibrator(config)
            kalman     = GazeKalmanFilter(config)
            saved      = False
            print("Calibration restarted")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()