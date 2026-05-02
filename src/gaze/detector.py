# src/gaze/detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional
from .face_data import FaceData


class FaceDetector:

    LEFT_IRIS        = 468
    RIGHT_IRIS       = 473
    LEFT_EYE_INNER   = 133
    LEFT_EYE_OUTER   = 33
    RIGHT_EYE_INNER  = 362
    RIGHT_EYE_OUTER  = 263
    LEFT_EYE_TOP     = 159
    LEFT_EYE_BOTTOM  = 145
    RIGHT_EYE_TOP    = 386
    RIGHT_EYE_BOTTOM = 374

    def __init__(self, config: dict):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config["gaze"]["confidence_threshold"],
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame: np.ndarray) -> Optional[FaceData]:
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        def to_px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        return FaceData(
            landmarks=lm,
            left_iris=to_px(self.LEFT_IRIS),
            right_iris=to_px(self.RIGHT_IRIS),
            left_eye_corners=(to_px(self.LEFT_EYE_INNER), to_px(self.LEFT_EYE_OUTER)),
            right_eye_corners=(to_px(self.RIGHT_EYE_INNER), to_px(self.RIGHT_EYE_OUTER)),
            confidence=1.0,
            frame_h=h,
            frame_w=w,
        )

    def close(self):
        self.face_mesh.close()