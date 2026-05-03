# src/gestures/recognizer.py

import time
import numpy as np
import pyautogui
from enum import Enum, auto
from .detector import HandData


class Gesture(Enum):
    NONE        = auto()
    LEFT_CLICK  = auto()
    RIGHT_CLICK = auto()
    DRAG_START  = auto()
    DRAG_END    = auto()


class GestureRecognizer:

    def __init__(self, config: dict):
        g = config["gestures"]
        self.click_threshold      = g["click_threshold"]
        self.drag_hold_seconds    = g["drag_hold_seconds"]
        self.two_finger_threshold = g["two_finger_threshold"]
        self.cooldown             = g["cooldown_seconds"]

        self._pinch_start_time  = None
        self._is_dragging       = False
        self._last_gesture_time = 0.0
        self._last_gesture      = Gesture.NONE

    def recognize(self, hand: HandData) -> Gesture:
        now       = time.time()
        hand_size = max(hand.hand_size, 1.0)

        thumb_index_dist  = self._dist(hand.thumb_tip, hand.index_tip)  / hand_size
        thumb_middle_dist = self._dist(hand.thumb_tip, hand.middle_tip) / hand_size

        is_index_pinch  = thumb_index_dist  < self.click_threshold
        is_middle_pinch = thumb_middle_dist < self.two_finger_threshold

        # ── Right click: index AND middle pinched ─────────────────────
        if is_index_pinch and is_middle_pinch:
            if self._can_trigger(now):
                self._last_gesture_time = now
                self._last_gesture      = Gesture.RIGHT_CLICK
                pyautogui.rightClick()
                return Gesture.RIGHT_CLICK

        # ── Left click / drag: only index pinched ─────────────────────
        elif is_index_pinch:
            if self._pinch_start_time is None:
                self._pinch_start_time = now

            pinch_duration = now - self._pinch_start_time

            if pinch_duration >= self.drag_hold_seconds and not self._is_dragging:
                self._is_dragging = True
                pyautogui.mouseDown()
                self._last_gesture = Gesture.DRAG_START
                return Gesture.DRAG_START

        else:
            # Pinch released
            if self._pinch_start_time is not None:
                pinch_duration = now - self._pinch_start_time

                if self._is_dragging:
                    pyautogui.mouseUp()
                    self._is_dragging      = False
                    self._pinch_start_time = None
                    self._last_gesture     = Gesture.DRAG_END
                    return Gesture.DRAG_END

                elif pinch_duration < self.drag_hold_seconds:
                    if self._can_trigger(now):
                        self._last_gesture_time = now
                        self._pinch_start_time  = None
                        self._last_gesture      = Gesture.LEFT_CLICK
                        pyautogui.click()
                        return Gesture.LEFT_CLICK

                self._pinch_start_time = None

        return Gesture.NONE

    def reset(self):
        """Call when hand leaves the frame."""
        if self._is_dragging:
            pyautogui.mouseUp()
            self._is_dragging = False
        self._pinch_start_time = None

    @property
    def is_dragging(self) -> bool:
        return self._is_dragging

    @property
    def last_gesture(self) -> Gesture:
        return self._last_gesture

    # ── Helpers ───────────────────────────────────────────────────────
    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _can_trigger(self, now: float) -> bool:
        return (now - self._last_gesture_time) >= self.cooldown