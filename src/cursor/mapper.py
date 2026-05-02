"""
src/cursor/mapper.py
Maps normalized gaze (0..1) → screen pixel coordinates (px, py).

No smoothing yet — that's Phase 7.
No calibration yet — that's Phase 8.
"""

import numpy as np


class GazeMapper:
    def __init__(self, config: dict):
        # Screen size
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
        self.pad_x = m["screen_padding_x"]
        self.pad_y = m["screen_padding_y"]

        # Usable screen area (avoids taskbar edges)
        self.usable_x_min = self.pad_x
        self.usable_x_max = self.screen_w - self.pad_x
        self.usable_y_min = self.pad_y
        self.usable_y_max = self.screen_h - self.pad_y

    def map(self, gaze_x: float, gaze_y: float) -> tuple[int, int]:
        """
        Convert raw gaze (0..1) to screen pixel (px, py).
        Clamps to usable screen area.
        """
        # Step 1: remap gaze range to 0..1
        nx = self._remap(gaze_x, self.gaze_x_min, self.gaze_x_max)
        ny = self._remap(gaze_y, self.gaze_y_min, self.gaze_y_max)

        # Step 2: invert X (webcam mirror flip)
        nx = 1.0 - nx

        # Step 3: scale to usable screen area
        px = int(self.usable_x_min + nx * (self.usable_x_max - self.usable_x_min))
        py = int(self.usable_y_min + ny * (self.usable_y_max - self.usable_y_min))

        # Step 4: hard clamp to screen bounds
        px = int(np.clip(px, 0, self.screen_w - 1))
        py = int(np.clip(py, 0, self.screen_h - 1))

        return px, py

    def update_range(self, gaze_x_min, gaze_x_max, gaze_y_min, gaze_y_max):
        """Called by calibration system (Phase 8) to update the mapping range."""
        self.gaze_x_min = gaze_x_min
        self.gaze_x_max = gaze_x_max
        self.gaze_y_min = gaze_y_min
        self.gaze_y_max = gaze_y_max

    def _remap(self, value: float, in_min: float, in_max: float) -> float:
        """Linearly remap value from [in_min, in_max] to [0, 1], clamped."""
        span = in_max - in_min
        if span < 1e-6:
            return 0.5
        return float(np.clip((value - in_min) / span, 0.0, 1.0))

    def _detect_screen(self) -> tuple[int, int]:
        """Auto-detect screen resolution on Windows."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()          # important for HiDPI screens
            w = user32.GetSystemMetrics(0)       # SM_CXSCREEN
            h = user32.GetSystemMetrics(1)       # SM_CYSCREEN
            print(f"[Mapper] Detected screen: {w}x{h}")
            return w, h
        except Exception as e:
            print(f"[Mapper] Screen detect failed ({e}), using config fallback")
            return 1920, 1080