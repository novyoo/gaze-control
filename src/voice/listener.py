# src/voice/listener.py
# Runs in a background thread.
# Puts recognized text into a queue for commander.py to process.

import threading
import queue
import logging
import speech_recognition as sr

logger = logging.getLogger(__name__)


class VoiceListener:

    def __init__(self, config: dict):
        v = config["voice"]
        self.language         = v["language"]
        self.energy_threshold = v["energy_threshold"]
        self.pause_threshold  = v["pause_threshold"]
        self.dynamic_energy   = v["dynamic_energy"]
        self.enabled          = v["enabled"]

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold    = self.energy_threshold
        self.recognizer.pause_threshold     = self.pause_threshold
        self.recognizer.dynamic_energy_threshold = self.dynamic_energy

        self.mic = sr.Microphone()

        # Thread-safe queue — listener puts text here, main loop reads it
        self._queue   = queue.Queue()
        self._thread  = None
        self._running = False

    def start(self):
        """Start listening in background thread."""
        if not self.enabled:
            logger.info("Voice listener disabled in config")
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._listen_loop,
            daemon=True,        # dies when main program exits
            name="VoiceListener"
        )
        self._thread.start()
        logger.info("Voice listener started")

    def stop(self):
        self._running = False
        logger.info("Voice listener stopped")

    def get_command(self) -> str | None:
        """
        Non-blocking — returns next command from queue or None.
        Call this every frame from your main loop.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _listen_loop(self):
        """Runs on background thread — continuously listens for speech."""
        logger.info("Adjusting for ambient noise...")

        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        logger.info("Listening for commands...")

        while self._running:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=5,           # wait max 5s for speech to start
                        phrase_time_limit=6  # max 6s per phrase
                    )

                text = self.recognizer.recognize_google(
                    audio,
                    language=self.language
                ).lower().strip()

                logger.info(f"Heard: '{text}'")
                self._queue.put(text)

            except sr.WaitTimeoutError:
                pass   # no speech heard — loop again
            except sr.UnknownValueError:
                pass   # couldn't understand audio — loop again
            except sr.RequestError as e:
                logger.error(f"Speech API error: {e}")
            except Exception as e:
                logger.error(f"Listener error: {e}")