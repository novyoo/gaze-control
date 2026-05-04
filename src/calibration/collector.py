# src/calibration/collector.py
# Collects raw gaze samples for each calibration point.

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CalibrationPoint:
    screen_x: float          # normalized 0..1 screen position
    screen_y: float
    gaze_samples: List[Tuple[float, float]] = field(default_factory=list)
    complete: bool = False

    def add_sample(self, gx: float, gy: float):
        self.gaze_samples.append((gx, gy))

    @property
    def mean_gaze(self) -> Tuple[float, float]:
        if not self.gaze_samples:
            return (0.5, 0.5)
        arr = np.array(self.gaze_samples)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


class GazeSampleCollector:

    # 9 calibration points — normalized screen positions
    CALIBRATION_POSITIONS = [
        (0.02,  0.02),   # top-left
        (0.5,   0.02),   # top-center
        (0.98,  0.02),   # top-right
        (0.02,  0.5),    # middle-left
        (0.5,   0.5),    # center
        (0.98,  0.5),    # middle-right
        (0.02,  0.98),   # bottom-left
        (0.5,   0.98),   # bottom-center
        (0.98,  0.98),   # bottom-right
    ]

    def __init__(self, config: dict):
        c = config["calibration"]
        self.samples_per_point = c["samples_per_point"]
        self.hold_seconds      = c["hold_seconds"]

        self.points     = [CalibrationPoint(sx, sy)
                           for sx, sy in self.CALIBRATION_POSITIONS]
        self.current    = 0
        self._hold_start = None
        self._collecting = False
        self._done       = False

    def update(self, gaze_x: float, gaze_y: float,
               is_blinking: bool) -> dict:
        """
        Call every frame with current gaze.
        Returns state dict for UI to render.
        """
        if self._done or self.current >= len(self.points):
            self._done = True
            return self._state("done")

        point = self.points[self.current]

        # Don't collect during blinks
        if is_blinking:
            self._hold_start = None
            return self._state("blink")

        # Start hold timer when gaze is somewhat stable
        if self._hold_start is None:
            self._hold_start  = time.time()
            self._collecting  = False

        elapsed = time.time() - self._hold_start

        # Brief delay before collecting — let eyes settle on dot
        if elapsed < 0.5:
            return self._state("waiting", elapsed / 0.5)

        # Collecting samples
        self._collecting = True
        point.add_sample(gaze_x, gaze_y)

        progress = len(point.gaze_samples) / self.samples_per_point

        if len(point.gaze_samples) >= self.samples_per_point:
            point.complete   = True
            self.current    += 1
            self._hold_start = None
            self._collecting = False

            if self.current >= len(self.points):
                self._done = True
                return self._state("done")

            return self._state("next")

        return self._state("collecting", progress)

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

    def _state(self, status: str, progress: float = 0.0) -> dict:
        return {
            "status":   status,
            "progress": progress,
            "index":    self.current,
            "total":    len(self.points),
        }