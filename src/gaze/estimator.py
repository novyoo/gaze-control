# src/gaze/estimator.py
# Head-compensated iris gaze estimator.
#
# Pipeline per frame:
#   1. Extract iris position in eye bounding box (0..1)
#   2. Estimate head rotation (yaw, pitch) from facial landmarks
#   3. Subtract head rotation contribution from iris position
#   4. Output compensated gaze (0..1) — pure eye movement only

import numpy as np
from .face_data import FaceData


class GazeEstimator:

    BLINK_EAR_THRESHOLD = 0.30

    # Full eye landmark sets for tight bounding box
    LEFT_EYE_ALL = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    ]
    RIGHT_EYE_ALL = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    ]

    # Stable facial landmarks for head pose
    NOSE_TIP    = 4
    CHIN        = 152
    LEFT_EYE_C  = 33
    RIGHT_EYE_C = 263
    LEFT_MOUTH  = 61
    RIGHT_MOUTH = 291

    # How strongly head rotation affects iris position
    # Tuned for top-center camera at 40-60cm
    HEAD_YAW_SCALE   = 0.012   # horizontal head influence per degree
    HEAD_PITCH_SCALE = 0.010   # vertical head influence per degree

    def __init__(self):
        self._last_gaze      = (0.5, 0.5)
        self._is_blinking    = False
        self._ear            = 0.0

        # Neutral head pose reference
        self._neutral_yaw    = None
        self._neutral_pitch  = None

        # For head stability detection (used by calibration)
        self._head_history   = []
        self._HEAD_WIN       = 6

    # ── Public API ────────────────────────────────────────────────────

    def estimate(self, face: FaceData) -> tuple[float, float]:
        """
        Main entry point. Returns compensated gaze (gx, gy) in 0..1.
        """
        self._ear         = self._eye_aspect_ratio(face)
        self._is_blinking = self._ear < self.BLINK_EAR_THRESHOLD

        if self._is_blinking:
            return self._last_gaze

        # 1. Raw iris position in eye bounding box
        raw_x, raw_y = self._raw_iris_position(face)

        # 2. Head rotation angles
        yaw, pitch = self._head_angles(face)

        # 3. Set neutral on first frame automatically
        if self._neutral_yaw is None:
            self._neutral_yaw  = yaw
            self._neutral_pitch = pitch

        # 4. Head rotation delta from neutral
        delta_yaw   = yaw   - self._neutral_yaw
        delta_pitch = pitch - self._neutral_pitch

        # 5. Subtract head rotation from iris
        # When head turns right (+yaw), iris appears to move left
        # We correct for this
        comp_x = raw_x - (delta_yaw   * self.HEAD_YAW_SCALE)
        comp_y = raw_y - (delta_pitch * self.HEAD_PITCH_SCALE)

        # 6. Clamp to valid range
        comp_x = float(np.clip(comp_x, 0.0, 1.0))
        comp_y = float(np.clip(comp_y, 0.0, 1.0))

        # 7. Track head for stability detection
        self._head_history.append((yaw, pitch))
        if len(self._head_history) > self._HEAD_WIN:
            self._head_history.pop(0)

        self._last_gaze = (comp_x, comp_y)
        return self._last_gaze

    def set_neutral(self, face: FaceData):
        """
        Call when user looks straight ahead at screen center.
        Resets the head pose reference point.
        """
        yaw, pitch          = self._head_angles(face)
        self._neutral_yaw   = yaw
        self._neutral_pitch = pitch
        print(f"[Estimator] Neutral set — yaw={yaw:.2f} pitch={pitch:.2f}")

    def calibrate_neutral(self, face: FaceData):
        """Alias for set_neutral — keeps backward compatibility."""
        self.set_neutral(face)

    def is_head_stable(self) -> bool:
        """
        Returns True if head has been still for last N frames.
        Used by calibration collector to decide when to sample.
        """
        if len(self._head_history) < self._HEAD_WIN:
            return False
        arr     = np.array(self._head_history)
        std_yaw   = float(np.std(arr[:, 0]))
        std_pitch = float(np.std(arr[:, 1]))
        return std_yaw < 0.8 and std_pitch < 0.8

    @property
    def is_blinking(self) -> bool:
        return self._is_blinking

    def ear_value(self, face: FaceData) -> float:
        return self._ear

    def head_angles(self, face: FaceData) -> tuple[float, float]:
        """Public access to head angles — used by calibration."""
        return self._head_angles(face)

    # ── Iris extraction ───────────────────────────────────────────────

    def _raw_iris_position(self, face: FaceData) -> tuple[float, float]:
        """
        Get iris position within eye bounding box.
        Returns (x, y) in 0..1 where:
            x: 0 = far left of eye,  1 = far right
            y: 0 = top of eye,       1 = bottom
        """
        lm   = face.landmarks
        w, h = face.frame_w, face.frame_h

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        left_pts  = np.array([px(i) for i in self.LEFT_EYE_ALL])
        right_pts = np.array([px(i) for i in self.RIGHT_EYE_ALL])

        left_x  = self._iris_x(face.left_iris,  left_pts)
        right_x = self._iris_x(face.right_iris, right_pts)
        left_y  = self._iris_y(face.left_iris,  left_pts)
        right_y = self._iris_y(face.right_iris, right_pts)

        # Average both eyes
        gaze_x = (left_x + right_x) / 2.0
        gaze_y = (left_y + right_y) / 2.0

        return float(gaze_x), float(gaze_y)

    def _iris_x(self, iris: np.ndarray,
                eye_pts: np.ndarray) -> float:
        x_min = float(np.min(eye_pts[:, 0]))
        x_max = float(np.max(eye_pts[:, 0]))
        span  = x_max - x_min
        if span < 1e-6:
            return 0.5
        return float(np.clip((iris[0] - x_min) / span, 0.0, 1.0))

    def _iris_y(self, iris: np.ndarray,
                eye_pts: np.ndarray) -> float:
        y_min = float(np.min(eye_pts[:, 1]))
        y_max = float(np.max(eye_pts[:, 1]))
        span  = y_max - y_min
        if span < 1e-6:
            return 0.5
        return float(np.clip((iris[1] - y_min) / span, 0.0, 1.0))

    # ── Head pose ─────────────────────────────────────────────────────

    def _head_angles(self, face: FaceData) -> tuple[float, float]:
        """
        Estimate yaw (left/right) and pitch (up/down) in degrees.
        Uses stable facial geometry — not affected by eye movement.
        """
        lm   = face.landmarks
        w, h = face.frame_w, face.frame_h

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        nose       = px(self.NOSE_TIP)
        chin       = px(self.CHIN)
        left_eye   = px(self.LEFT_EYE_C)
        right_eye  = px(self.RIGHT_EYE_C)

        eye_mid    = (left_eye + right_eye) / 2.0
        face_width = np.linalg.norm(right_eye - left_eye)
        face_height= np.linalg.norm(chin - eye_mid)

        if face_width < 1e-6 or face_height < 1e-6:
            return 0.0, 0.0

        # Yaw: how far nose deviates horizontally from eye center
        yaw   = (nose[0] - eye_mid[0]) / face_width * 90.0

        # Pitch: how far nose deviates vertically from face midpoint
        face_mid_y = (eye_mid[1] + chin[1]) / 2.0
        pitch = (nose[1] - face_mid_y) / face_height * 90.0

        return float(yaw), float(pitch)

    # ── EAR ───────────────────────────────────────────────────────────

    def _eye_aspect_ratio(self, face: FaceData) -> float:
        lm   = face.landmarks
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