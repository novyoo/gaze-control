# src/calibration/collector.py
# Strict sample collector — only collects when:
#   1. Dot has been shown long enough (get ready time)
#   2. Head is stable (no movement)
#   3. Iris position is stable (fixating, not drifting)
#   4. Not blinking

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class CalibrationPoint:
    screen_x: float
    screen_y: float
    # Stores RAW iris positions (before compensation)
    # Calibration maps raw iris → screen directly
    iris_samples: List[Tuple[float, float]] = field(default_factory=list)
    head_samples: List[Tuple[float, float]] = field(default_factory=list)
    complete: bool = False

    def add_sample(self, iris_x: float, iris_y: float,
                   head_yaw: float, head_pitch: float):
        self.iris_samples.append((iris_x, iris_y))
        self.head_samples.append((head_yaw, head_pitch))

    @property
    def mean_gaze(self) -> Tuple[float, float]:
        """
        Returns median iris position for this calibration point.
        Used by calibrator to fit the regression.
        """
        if not self.iris_samples:
            return (0.5, 0.5)
        arr = np.array(self.iris_samples)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    @property
    def mean_head(self) -> Tuple[float, float]:
        if not self.head_samples:
            return (0.0, 0.0)
        arr = np.array(self.head_samples)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


class GazeSampleCollector:

    CALIBRATION_POSITIONS = [
        (0.02, 0.02),   # top-left
        (0.5,  0.02),   # top-center
        (0.98, 0.02),   # top-right
        (0.02, 0.5),    # middle-left
        (0.5,  0.5),    # center
        (0.98, 0.5),    # middle-right
        (0.02, 0.98),   # bottom-left
        (0.5,  0.98),   # bottom-center
        (0.98, 0.98),   # bottom-right
    ]

    def __init__(self, config: dict):
        c = config["calibration"]
        self.samples_per_point = c["samples_per_point"]

        self.points = [
            CalibrationPoint(sx, sy)
            for sx, sy in self.CALIBRATION_POSITIONS
        ]
        self.current         = 0
        self._dot_shown_at   = None
        self._hold_start     = None
        self._done           = False

        # Stability windows
        self._recent_iris    = []
        self._IRIS_WIN       = 10
        self._IRIS_STD_THR   = 0.018   # max iris std to be stable

        self._SHOW_DELAY     = 2.0     # seconds before collecting starts
        self._HOLD_DELAY     = 0.5     # seconds of stability before sampling

    def update(self, iris_x: float, iris_y: float,
               head_yaw: float, head_pitch: float,
               is_blinking: bool,
               head_stable: bool) -> dict:
        """
        Call every frame.
        iris_x/y  — raw iris position (not compensated)
        head_yaw/pitch — current head angles
        is_blinking — from estimator
        head_stable — from estimator.is_head_stable()
        """
        if self._done or self.current >= len(self.points):
            self._done = True
            return self._state("done")

        # Record when dot first shown
        if self._dot_shown_at is None:
            self._dot_shown_at = time.time()
            self._hold_start   = None
            self._recent_iris  = []

        now     = time.time()
        elapsed = now - self._dot_shown_at

        # Phase 1 — show delay, give user time to find dot
        if elapsed < self._SHOW_DELAY:
            return self._state("get_ready",
                               elapsed / self._SHOW_DELAY)

        # Phase 2 — skip if blinking
        if is_blinking:
            self._hold_start = None
            return self._state("blink")

        # Phase 3 — wait for head to be stable
        if not head_stable:
            self._hold_start = None
            return self._state("stabilizing")

        # Phase 4 — wait for iris to be stable
        self._recent_iris.append((iris_x, iris_y))
        if len(self._recent_iris) > self._IRIS_WIN:
            self._recent_iris.pop(0)

        if not self._iris_stable():
            self._hold_start = None
            return self._state("stabilizing")

        # Phase 5 — hold timer
        if self._hold_start is None:
            self._hold_start = now

        hold_elapsed = now - self._hold_start
        if hold_elapsed < self._HOLD_DELAY:
            return self._state("hold_steady",
                               hold_elapsed / self._HOLD_DELAY)

        # Phase 6 — collect sample
        point = self.points[self.current]
        point.add_sample(iris_x, iris_y, head_yaw, head_pitch)

        progress = len(point.iris_samples) / self.samples_per_point

        if len(point.iris_samples) >= self.samples_per_point:
            point.complete     = True
            self.current      += 1
            self._dot_shown_at = None
            self._hold_start   = None
            self._recent_iris  = []

            if self.current >= len(self.points):
                self._done = True
                return self._state("done")

            return self._state("next")

        return self._state("collecting", progress)

    def _iris_stable(self) -> bool:
        if len(self._recent_iris) < self._IRIS_WIN:
            return False
        arr   = np.array(self._recent_iris)
        std_x = float(np.std(arr[:, 0]))
        std_y = float(np.std(arr[:, 1]))
        return std_x < self._IRIS_STD_THR and std_y < self._IRIS_STD_THR

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def current_point(self) -> CalibrationPoint:
        if self.current < len(self.points):
            return self.points[self.current]
        return self.points[-1]

    @property
    def completed_points(self) -> list:
        return [p for p in self.points if p.complete]

    def _state(self, status: str,
               progress: float = 0.0) -> dict:
        return {
            "status":   status,
            "progress": progress,
            "index":    self.current,
            "total":    len(self.points),
        }