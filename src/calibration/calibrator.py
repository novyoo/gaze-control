# src/calibration/calibrator.py
# Fits gaze → screen mapping from collected samples.
# Saves and loads calibration data.

import json
import logging
import numpy as np
from typing import Optional
from .collector import GazeSampleCollector

logger = logging.getLogger(__name__)


class GazeCalibrator:

    def __init__(self, config: dict):
        self.save_path = config["calibration"]["save_path"]
        self._coeffs_x = None   # polynomial coefficients for X
        self._coeffs_y = None   # polynomial coefficients for Y
        self._is_calibrated = False

    def fit(self, collector: GazeSampleCollector) -> bool:
        """
        Fit a polynomial mapping from gaze → screen coords
        using completed calibration points.
        """
        points = collector.completed_points
        if len(points) < 4:
            logger.error("Need at least 4 calibration points")
            return False

        # Build arrays
        gaze_x   = np.array([p.mean_gaze[0] for p in points])
        gaze_y   = np.array([p.mean_gaze[1] for p in points])
        screen_x = np.array([p.screen_x     for p in points])
        screen_y = np.array([p.screen_y     for p in points])

        # Fit degree-2 polynomial for each axis
        # screen_x = f(gaze_x, gaze_y)
        # screen_y = g(gaze_x, gaze_y)
        # Features: [1, gx, gy, gx^2, gy^2, gx*gy]
        features = self._build_features(gaze_x, gaze_y)

        try:
            self._coeffs_x, _, _, _ = np.linalg.lstsq(
                features, screen_x, rcond=None)
            self._coeffs_y, _, _, _ = np.linalg.lstsq(
                features, screen_y, rcond=None)
            self._is_calibrated = True
            logger.info("Calibration fit successful")
            return True
        except Exception as e:
            logger.error(f"Calibration fit failed: {e}")
            return False

    def map(self, gaze_x: float,
            gaze_y: float) -> tuple[float, float]:
        """
        Map raw gaze to normalized screen position using
        fitted polynomial.
        Falls back to linear if not calibrated.
        """
        if not self._is_calibrated:
            return gaze_x, gaze_y

        features = self._build_features(
            np.array([gaze_x]),
            np.array([gaze_y])
        )

        sx = float(np.clip(features @ self._coeffs_x, 0.0, 1.0))
        sy = float(np.clip(features @ self._coeffs_y, 0.0, 1.0))
        return sx, sy

    def save(self) -> bool:
        """Save calibration coefficients to JSON."""
        if not self._is_calibrated:
            return False
        try:
            data = {
                "coeffs_x": self._coeffs_x.tolist(),
                "coeffs_y": self._coeffs_y.tolist(),
                "version":  1,
            }
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Calibration saved to {self.save_path}")
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def load(self) -> bool:
        """Load calibration from JSON. Returns True if successful."""
        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
            self._coeffs_x      = np.array(data["coeffs_x"])
            self._coeffs_y      = np.array(data["coeffs_y"])
            self._is_calibrated = True
            logger.info(f"Calibration loaded from {self.save_path}")
            return True
        except FileNotFoundError:
            logger.info("No calibration file found — run calibration first")
            return False
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def _build_features(self, gx: np.ndarray,
                        gy: np.ndarray) -> np.ndarray:
        """
        Build polynomial feature matrix.
        [1, gx, gy, gx^2, gy^2, gx*gy]
        """
        ones = np.ones_like(gx)
        return np.column_stack([
            ones, gx, gy,
            gx ** 2, gy ** 2,
            gx * gy
        ])