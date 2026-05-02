"""
src/gaze/estimator.py
Estimates gaze direction from iris position relative to eye corners.
Output: normalized (gx, gy) in range [0, 1] — raw, no smoothing yet.
"""

import numpy as np
from .detector import FaceData


class GazeEstimator:
    def estimate(self, face: FaceData) -> tuple[float, float]:
        """
        Returns (gaze_x, gaze_y) normalized 0..1.
        
        Method: iris position as fraction of eye width/height.
        Left eye and right eye averaged for stability.
        """
        left_gx  = self._iris_ratio(face.left_iris,  face.left_eye_corners)
        right_gx = self._iris_ratio(face.right_iris, face.right_eye_corners)

        # Average both eyes — more robust than using one
        gaze_x = (left_gx + right_gx) / 2.0

        # Vertical: use left iris Y relative to eye bounding box
        # We use left_eye_corners inner/outer to approximate the eye center Y
        eye_center_y = (face.left_eye_corners[0][1] + face.left_eye_corners[1][1]) / 2
        eye_h_approx = abs(face.left_iris[1] - eye_center_y) * 4  # rough estimate
        gaze_y = np.clip(
            (face.left_iris[1] - (eye_center_y - eye_h_approx / 2)) / max(eye_h_approx, 1),
            0.0, 1.0
        )

        return float(gaze_x), float(gaze_y)

    def _iris_ratio(self, iris: np.ndarray, corners: tuple) -> float:
        """
        Where is the iris between the two eye corners?
        Returns 0.0 (far left) to 1.0 (far right).
        """
        inner, outer = corners
        eye_vec  = outer - inner
        iris_vec = iris  - inner
        eye_len  = np.linalg.norm(eye_vec)

        if eye_len < 1e-6:
            return 0.5  # degenerate case — return center

        return float(np.clip(np.dot(iris_vec, eye_vec) / (eye_len ** 2), 0.0, 1.0))