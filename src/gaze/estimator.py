# src/gaze/estimator.py

import numpy as np
from .face_data import FaceData


class GazeEstimator:

    BLINK_EAR_THRESHOLD = 0.30

    def __init__(self):
        self._last_gaze = (0.5, 0.5)
        self._is_blinking = False
        self._ear = 0.0

    def estimate(self, face: FaceData) -> tuple[float, float]:
        self._ear = self._eye_aspect_ratio(face)
        self._is_blinking = self._ear < self.BLINK_EAR_THRESHOLD

        if self._is_blinking:
            return self._last_gaze

        gaze_x = self._estimate_horizontal(face)
        gaze_y = self._estimate_vertical(face)
        self._last_gaze = (gaze_x, gaze_y)
        return self._last_gaze

    @property
    def is_blinking(self) -> bool:
        return self._is_blinking

    def ear_value(self, face: FaceData) -> float:
        return self._ear

    def _estimate_horizontal(self, face: FaceData) -> float:
        left_gx  = self._iris_ratio(face.left_iris,  face.left_eye_corners)
        right_gx = self._iris_ratio(face.right_iris, face.right_eye_corners)
        return float((left_gx + right_gx) / 2.0)

    def _estimate_vertical(self, face: FaceData) -> float:
        lm = face.landmarks
        w, h = face.frame_w, face.frame_h

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        left_y  = self._iris_y_ratio(face.left_iris,  px(159), px(145))
        right_y = self._iris_y_ratio(face.right_iris, px(386), px(374))
        return float((left_y + right_y) / 2.0)

    def _eye_aspect_ratio(self, face: FaceData) -> float:
        lm = face.landmarks
        w, h = face.frame_w, face.frame_h

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        def ear(top, bottom, inner, outer):
            eye_h = np.linalg.norm(px(top)   - px(bottom))
            eye_w = np.linalg.norm(px(inner) - px(outer))
            return eye_h / max(eye_w, 1e-6)

        left_ear  = ear(159, 145, 133, 33)
        right_ear = ear(386, 374, 362, 263)
        return (left_ear + right_ear) / 2.0

    def _iris_ratio(self, iris: np.ndarray, corners: tuple) -> float:
        inner, outer = corners
        eye_vec  = outer - inner
        iris_vec = iris  - inner
        eye_len  = np.linalg.norm(eye_vec)
        if eye_len < 1e-6:
            return 0.5
        return float(np.clip(np.dot(iris_vec, eye_vec) / (eye_len ** 2), 0.0, 1.0))

    def _iris_y_ratio(self, iris: np.ndarray,
                      top: np.ndarray, bottom: np.ndarray) -> float:
        eye_h = bottom[1] - top[1]
        if abs(eye_h) < 1e-6:
            return 0.5
        return float(np.clip((iris[1] - top[1]) / eye_h, 0.0, 1.0))