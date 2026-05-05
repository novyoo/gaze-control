# src/cursor/mapper.py

import numpy as np


class GazeMapper:

    def __init__(self, config: dict):
        if config["screen"]["auto_detect"]:
            self.screen_w, self.screen_h = self._detect_screen()
        else:
            self.screen_w = config["screen"]["width"]
            self.screen_h = config["screen"]["height"]

        m = config["mapping"]
        self.gaze_x_min = m["gaze_x_min"]
        self.gaze_x_max = m["gaze_x_max"]
        self.gaze_y_min = m["gaze_y_min"]
        self.gaze_y_max = m["gaze_y_max"]
        self.pad_x      = m["screen_padding_x"]
        self.pad_y      = m["screen_padding_y"]

        self.usable_x_min = self.pad_x
        self.usable_x_max = self.screen_w - self.pad_x
        self.usable_y_min = self.pad_y
        self.usable_y_max = self.screen_h - self.pad_y

        self._calibrator = None

    def map(self, gaze_x: float, gaze_y: float) -> tuple[int, int]:
        """
        Convert gaze (0..1) → screen pixel (px, py).
        Uses calibration if available, otherwise linear mapping.
        """
        if self._calibrator is not None and self._calibrator.is_calibrated:
            # Calibration already maps gaze → normalized screen
            nx, ny = self._calibrator.map(gaze_x, gaze_y)
            nx = float(np.clip(nx, 0.0, 1.0))
            ny = float(np.clip(ny, 0.0, 1.0))
            # Still invert X — camera is mirrored
            nx = 1.0 - nx
        else:
            # Linear mapping using measured gaze range
            nx = self._remap(gaze_x, self.gaze_x_min, self.gaze_x_max)
            ny = self._remap(gaze_y, self.gaze_y_min, self.gaze_y_max)
            # Invert X — camera is mirrored
            nx = 1.0 - nx

        px = int(self.usable_x_min + nx * (self.usable_x_max - self.usable_x_min))
        py = int(self.usable_y_min + ny * (self.usable_y_max - self.usable_y_min))
        px = int(np.clip(px, 0, self.screen_w - 1))
        py = int(np.clip(py, 0, self.screen_h - 1))
        return px, py

    def apply_calibration(self, calibrator) -> None:
        """Load calibration into mapper."""
        if calibrator.is_calibrated:
            self._calibrator = calibrator
            print("[Mapper] Calibration applied")

    def update_range(self, gaze_x_min, gaze_x_max,
                     gaze_y_min, gaze_y_max):
        """Manually update mapping range — used by calibration system."""
        self.gaze_x_min = gaze_x_min
        self.gaze_x_max = gaze_x_max
        self.gaze_y_min = gaze_y_min
        self.gaze_y_max = gaze_y_max

    def _remap(self, value: float,
               in_min: float, in_max: float) -> float:
        span = in_max - in_min
        if span < 1e-6:
            return 0.5
        return float(np.clip((value - in_min) / span, 0.0, 1.0))

    def _detect_screen(self) -> tuple[int, int]:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            print(f"[Mapper] Detected screen: {w}x{h}")
            return w, h
        except Exception as e:
            print(f"[Mapper] Screen detect failed ({e}), using fallback")
            return 1920, 1080