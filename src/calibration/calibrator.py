# src/calibration/calibrator.py

import json
import logging
import numpy as np
from typing import Optional
from .collector import GazeSampleCollector

logger = logging.getLogger(__name__)


class GazeCalibrator:

    def __init__(self, config: dict):
        self.save_path      = config["calibration"]["save_path"]
        self._coeffs_x      = None
        self._coeffs_y      = None
        self._is_calibrated = False

    def fit(self, collector: GazeSampleCollector) -> bool:
        points = collector.completed_points
        if len(points) < 4:
            logger.error("Need at least 4 calibration points")
            return False

        # Build input features from raw iris + head pose
        iris_x    = np.array([p.mean_gaze[0]  for p in points])
        iris_y    = np.array([p.mean_gaze[1]  for p in points])
        head_yaw  = np.array([p.mean_head[0]  for p in points])
        head_pitch= np.array([p.mean_head[1]  for p in points])
        screen_x  = np.array([p.screen_x      for p in points])
        screen_y  = np.array([p.screen_y      for p in points])

        # Feature vector:
        # [1, ix, iy, yaw, pitch, ix^2, iy^2, ix*iy, ix*yaw, iy*pitch]
        features = self._build_features(
            iris_x, iris_y, head_yaw, head_pitch)

        try:
            self._coeffs_x, _, _, _ = np.linalg.lstsq(
                features, screen_x, rcond=None)
            self._coeffs_y, _, _, _ = np.linalg.lstsq(
                features, screen_y, rcond=None)
            self._is_calibrated = True

            # Log fit quality
            pred_x = features @ self._coeffs_x
            pred_y = features @ self._coeffs_y
            err_x  = float(np.mean(np.abs(pred_x - screen_x)))
            err_y  = float(np.mean(np.abs(pred_y - screen_y)))
            logger.info(f"Calibration fit — MAE x:{err_x:.3f} y:{err_y:.3f}")
            print(f"[Calibrator] Fit error: x={err_x:.3f} y={err_y:.3f} "
                  f"(lower is better, <0.05 is good)")
            return True

        except Exception as e:
            logger.error(f"Calibration fit failed: {e}")
            return False

    def map(self, iris_x: float, iris_y: float,
            head_yaw: float = 0.0,
            head_pitch: float = 0.0) -> tuple[float, float]:
        """Map iris + head pose to normalized screen position."""
        if not self._is_calibrated:
            return iris_x, iris_y

        features = self._build_features(
            np.array([iris_x]),
            np.array([iris_y]),
            np.array([head_yaw]),
            np.array([head_pitch])
        )

        sx = float(np.clip(features @ self._coeffs_x, 0.0, 1.0))
        sy = float(np.clip(features @ self._coeffs_y, 0.0, 1.0))
        return sx, sy

    def save(self) -> bool:
        if not self._is_calibrated:
            return False
        try:
            data = {
                "coeffs_x": self._coeffs_x.tolist(),
                "coeffs_y": self._coeffs_y.tolist(),
                "version":  2,
            }
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Calibrator] Saved to {self.save_path}")
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def load(self) -> bool:
        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
            self._coeffs_x      = np.array(data["coeffs_x"])
            self._coeffs_y      = np.array(data["coeffs_y"])
            self._is_calibrated = True
            print(f"[Calibrator] Loaded from {self.save_path}")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def _build_features(self, ix: np.ndarray, iy: np.ndarray,
                        yaw: np.ndarray,
                        pitch: np.ndarray) -> np.ndarray:
        ones = np.ones_like(ix)
        return np.column_stack([
            ones,
            ix, iy,
            yaw, pitch,
            ix ** 2, iy ** 2,
            ix * iy,
            ix * yaw,
            iy * pitch,
        ])