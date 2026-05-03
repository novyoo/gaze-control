# src/gestures/detector.py

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HandData:
    landmarks: list          # all 21 landmarks normalized 0..1
    thumb_tip: np.ndarray    # pixel coords
    index_tip: np.ndarray
    middle_tip: np.ndarray
    ring_tip: np.ndarray
    pinky_tip: np.ndarray
    wrist: np.ndarray
    index_mcp: np.ndarray    # base of index finger — used for hand size reference
    frame_h: int
    frame_w: int

    @property
    def hand_size(self) -> float:
        """Distance from wrist to index MCP — used to normalize distances."""
        return float(np.linalg.norm(self.index_mcp - self.wrist))


class HandDetector:

    # Landmark indices
    WRIST      = 0
    THUMB_TIP  = 4
    INDEX_TIP  = 8
    MIDDLE_TIP = 12
    RING_TIP   = 16
    PINKY_TIP  = 20
    INDEX_MCP  = 5

    def __init__(self, config: dict):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

    def process(self, frame: np.ndarray) -> Optional[HandData]:
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0].landmark

        def to_px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        return HandData(
            landmarks=lm,
            thumb_tip=to_px(self.THUMB_TIP),
            index_tip=to_px(self.INDEX_TIP),
            middle_tip=to_px(self.MIDDLE_TIP),
            ring_tip=to_px(self.RING_TIP),
            pinky_tip=to_px(self.PINKY_TIP),
            wrist=to_px(self.WRIST),
            index_mcp=to_px(self.INDEX_MCP),
            frame_h=h,
            frame_w=w,
        )

    def close(self):
        self.hands.close()