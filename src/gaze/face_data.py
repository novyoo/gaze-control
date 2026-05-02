# src/gaze/face_data.py

import numpy as np
from dataclasses import dataclass


@dataclass
class FaceData:
    landmarks: list
    left_iris: np.ndarray
    right_iris: np.ndarray
    left_eye_corners: tuple
    right_eye_corners: tuple
    confidence: float
    frame_h: int
    frame_w: int