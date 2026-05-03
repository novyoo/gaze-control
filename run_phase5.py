# run_phase5.py
# Phase 5: hand gesture test
# Shows camera feed with gesture overlay
# Press Q to quit

import cv2
import time
import yaml
import numpy as np
from src.gestures.detector   import HandDetector
from src.gestures.recognizer import GestureRecognizer, Gesture


def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


GESTURE_COLORS = {
    Gesture.NONE:        (160, 160, 160),
    Gesture.LEFT_CLICK:  (0,   255, 100),
    Gesture.RIGHT_CLICK: (0,   180, 255),
    Gesture.DRAG_START:  (255, 180,   0),
    Gesture.DRAG_END:    (255, 100,   0),
}

def draw_hand(frame, hand):
    tips   = [hand.thumb_tip, hand.index_tip, hand.middle_tip,
              hand.ring_tip,  hand.pinky_tip]
    colors = [(255,255,255), (0,255,100), (0,200,255),
              (180,100,255), (255,100,100)]

    for tip, color in zip(tips, colors):
        cv2.circle(frame, tuple(tip.astype(int)), 8, color, -1)

    cv2.circle(frame, tuple(hand.wrist.astype(int)), 6, (200,200,200), 2)

    cv2.line(frame,
             tuple(hand.thumb_tip.astype(int)),
             tuple(hand.index_tip.astype(int)),
             (0, 255, 100), 1)
    cv2.line(frame,
             tuple(hand.thumb_tip.astype(int)),
             tuple(hand.middle_tip.astype(int)),
             (0, 180, 255), 1)


def draw_hud(frame, hand, gesture, fps, recognizer):
    h, w = frame.shape[:2]
    color = GESTURE_COLORS.get(gesture, (255, 255, 255))

    # ── Compute hand metrics ──────────────────────────────────────────
    if hand:
        hs         = max(hand.hand_size, 1.0)
        norm_pinch = np.linalg.norm(hand.thumb_tip - hand.index_tip) / hs
        gap        = (hand.wrist[1] - hand.thumb_tip[1]) / hs
    else:
        norm_pinch = 0.0
        gap        = 0.0

    lines = [
        (f"FPS:        {fps:.1f}",                                   (255, 255,   0)),
        (f"Gesture:    {gesture.name}",                               GESTURE_COLORS.get(gesture, (255,255,255))),
        (f"Dragging:   {'YES' if recognizer.is_dragging else 'no'}", (255,180,0) if recognizer.is_dragging else (160,160,160)),
        (f"Norm pinch: {norm_pinch:.3f}",                             (255, 255, 255)),
        (f"Thumb gap:  {gap:.3f}",                                    (0,255,100) if gap > 0.3 else (0,180,255) if gap < -0.3 else (160,160,160)),
        (f"Thumbs UP"  if gap > 0.3  else
        f"Thumbs DOWN" if gap < -0.3 else
        f"Hand flat",                                                (0,255,100) if gap > 0.3 else (0,180,255) if gap < -0.3 else (160,160,160)),
    ]

    for i, (text, c) in enumerate(lines):
        cv2.putText(frame, text, (20, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, c, 1)

    # ── Big gesture flash in center ───────────────────────────────────
    if gesture != Gesture.NONE:
        cv2.putText(frame, gesture.name,
                    (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

    # ── Scroll guide arrows ───────────────────────────────────────────
    cv2.putText(frame, "^ move hand UP   = scroll up",
                (20, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)
    cv2.putText(frame, "v move hand DOWN = scroll down",
                (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)
    cv2.putText(frame, "Q = quit",
                (w - 100, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)


def main():
    config     = load_config()
    detector   = HandDetector(config)
    recognizer = GestureRecognizer(config)

    cap = cv2.VideoCapture(config["camera"]["index"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS,          config["camera"]["fps"])

    prev_time = time.time()
    fps       = 0.0
    gesture   = Gesture.NONE

    print("Phase 5 running — show your hand to the camera")
    print("Index pinch         = LEFT CLICK")
    print("Index+middle pinch  = RIGHT CLICK")
    print("Hold index pinch    = DRAG")
    print("Move hand UP        = SCROLL UP")
    print("Move hand DOWN      = SCROLL DOWN")
    print("Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hand  = detector.process(frame)

        now       = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        if hand:
            gesture = recognizer.recognize(hand)
            draw_hand(frame, hand)
        else:
            recognizer.reset()
            gesture = Gesture.NONE

        draw_hud(frame, hand, gesture, fps, recognizer)

        cv2.imshow("Phase 5 — Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()