import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FaceData:
    """All the face info we need per frame."""
    landmarks: list          # All 478 landmark (x, y, z) — normalized 0..1
    left_iris: np.ndarray    # (x, y) pixel coords of left iris center
    right_iris: np.ndarray   # (x, y) pixel coords of right iris center
    left_eye_corners: tuple  # ( (x,y), (x,y) ) — inner, outer
    right_eye_corners: tuple
    confidence: float        # Detection confidence 0..1
    frame_h: int
    frame_w: int


class FaceDetector:
    # Iris landmark indices (MediaPipe spec)
    LEFT_IRIS  = 468
    RIGHT_IRIS = 473

    # Eye corner indices
    LEFT_EYE_INNER  = 133   # closest to nose
    LEFT_EYE_OUTER  = 33    # closest to ear
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    def __init__(self, config: dict):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # REQUIRED for iris landmarks (468-477)
            min_detection_confidence=config["gaze"]["confidence_threshold"],
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Run face mesh on one frame.
        Returns FaceData if a face is found, None otherwise.
        """
        h, w = frame.shape[:2]

        # MediaPipe needs RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # small perf gain
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark  # first face

        def to_px(idx):
            """Convert normalized landmark to pixel coords."""
            return np.array([lm[idx].x * w, lm[idx].y * h])

        return FaceData(
            landmarks=lm,
            left_iris=to_px(self.LEFT_IRIS),
            right_iris=to_px(self.RIGHT_IRIS),
            left_eye_corners=(to_px(self.LEFT_EYE_INNER), to_px(self.LEFT_EYE_OUTER)),
            right_eye_corners=(to_px(self.RIGHT_EYE_INNER), to_px(self.RIGHT_EYE_OUTER)),
            confidence=1.0,   # FaceMesh doesn't expose per-landmark confidence
            frame_h=h,
            frame_w=w,
        )

    def close(self):
        self.face_mesh.close()