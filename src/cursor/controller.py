# src/cursor/controller.py

import pyautogui
import numpy as np


# Safety config — do this before anything else
pyautogui.FAILSAFE   = True   # move mouse to top-left corner to emergency stop
pyautogui.PAUSE      = 0.0    # remove built-in delay (we handle timing ourselves)


class CursorController:

    def __init__(self, config: dict):
        c = config["cursor"]

        # Smoothing: 0.0 = never moves, 1.0 = raw/jumpy
        # 0.12 is a good starting point — adjust in settings.yaml
        self.smooth_factor  = c["smooth_factor"]

        # Deadzone: ignore movements smaller than this (pixels)
        self.deadzone_radius = c["deadzone_radius"]

        # Get screen size
        self.screen_w, self.screen_h = pyautogui.size()

        # Internal state
        self._smooth_x = float(self.screen_w  // 2)
        self._smooth_y = float(self.screen_h // 2)
        self._enabled  = True

    def move(self, target_x: int, target_y: int) -> tuple[int, int]:
        """
        Smoothly move cursor toward (target_x, target_y).
        Returns the actual position moved to.
        """
        if not self._enabled:
            return int(self._smooth_x), int(self._smooth_y)

        # Exponential moving average (EMA)
        self._smooth_x = (self.smooth_factor * target_x
                          + (1 - self.smooth_factor) * self._smooth_x)
        self._smooth_y = (self.smooth_factor * target_y
                          + (1 - self.smooth_factor) * self._smooth_y)

        final_x = int(self._smooth_x)
        final_y = int(self._smooth_y)

        # Deadzone check — only move if far enough from current position
        cur_x, cur_y = pyautogui.position()
        dist = np.hypot(final_x - cur_x, final_y - cur_y)

        if dist > self.deadzone_radius:
            pyautogui.moveTo(final_x, final_y)

        return final_x, final_y

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def get_smooth_pos(self) -> tuple[int, int]:
        return int(self._smooth_x), int(self._smooth_y)