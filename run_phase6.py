# run_phase6.py
# Phase 6: voice command test — no camera needed
# Just speaks commands and shows results in terminal + overlay window
# Press Q to quit

import cv2
import time
import yaml
import numpy as np
import logging
from src.voice.listener  import VoiceListener
from src.voice.commander import VoiceCommander

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def main():
    config   = load_config()
    listener = VoiceListener(config)
    commander = VoiceCommander()

    # Start mic in background thread
    listener.start()

    sw, sh = 800, 400
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

    # Keep last 6 command results for display
    history = []

    print("\n=== Phase 6 Voice Control ===")
    print("Say: 'search python tutorials'")
    print("Say: 'open notepad'")
    print("Say: 'screenshot'")
    print("Say: 'volume up' / 'volume down'")
    print("Say: 'scroll up' / 'scroll down'")
    print("Say: 'stop listening' / 'start listening'")
    print("Press Q in window to quit\n")

    prev_time = time.time()
    fps       = 0.0

    while True:
        # ── Check for new voice command ────────────────────────────
        text = listener.get_command()
        if text:
            result = commander.process(text)
            if result:
                status = "✓" if result.success else "✗"
                entry  = f"{status} '{result.command}' → {result.action}"
                history.append((entry, result.success))
                if len(history) > 6:
                    history.pop(0)
                print(f"[VOICE] {entry}")

        # ── Draw UI ────────────────────────────────────────────────
        canvas[:] = 18

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # Title
        cv2.putText(canvas, "VOICE CONTROL — Phase 6",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 1)

        # Status
        status_color = (0, 100, 255) if commander.is_paused else (0, 255, 100)
        status_text  = "PAUSED — say 'start listening'" if commander.is_paused else "LISTENING..."
        cv2.putText(canvas, status_text,
                    (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 1)

        # FPS
        cv2.putText(canvas, f"FPS: {fps:.1f}",
                    (sw - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        # Command history
        cv2.putText(canvas, "Command history:",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)

        for i, (entry, success) in enumerate(reversed(history)):
            color = (0, 255, 100) if success else (0, 100, 255)
            cv2.putText(canvas, entry,
                        (20, 162 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        # Commands reference
        commands = [
            "search [query]       → opens browser",
            "type [text]          → types at cursor",
            "open [any app name]",
            "screenshot",
            "volume up/down [1-50]",
            "scroll up/down [1-50]",
            "stop / start listening",
]
        cv2.putText(canvas, "Commands:",
                    (sw - 260, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)
        for i, cmd in enumerate(commands):
            cv2.putText(canvas, cmd,
                        (sw - 260, 158 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)

        cv2.putText(canvas, "Q = quit",
                    (20, sh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

        cv2.imshow("Voice Control — Phase 6", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    listener.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()