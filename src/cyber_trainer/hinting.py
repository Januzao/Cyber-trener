from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pyttsx3

from .rules_deadlift import Hint


class TTSWorker(threading.Thread):
    """
    Background TTS so main loop never blocks.
    """
    def __init__(self) -> None:
        super().__init__(daemon=True, name="tts-worker")
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()

    def run(self) -> None:
        engine = pyttsx3.init()
        # You can tune voice rate/volume here if you want:
        # engine.setProperty("rate", 175)
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                # if TTS fails, ignore to keep app running
                pass

    def stop(self) -> None:
        self._stop.set()

    def speak(self, text: str) -> None:
        # Drop if queue too large to avoid backlog
        if self.q.qsize() > 3:
            return
        self.q.put(text)


class HintManager:
    """
    Global rate limit + per-issue cooldown + (optional) voice output.
    """
    def __init__(self, enable_voice: bool, max_hints_per_sec: float, per_issue_cooldown_sec: float) -> None:
        self.enable_voice = enable_voice
        self.min_interval = 1.0 / max(0.1, max_hints_per_sec)
        self.cooldown = per_issue_cooldown_sec

        self._last_global_t = 0.0
        self._last_issue_t: Dict[str, float] = {}

        self._tts: Optional[TTSWorker] = None
        if self.enable_voice:
            self._tts = TTSWorker()
            self._tts.start()

    def shutdown(self) -> None:
        if self._tts is not None:
            self._tts.stop()

    def allow(self, hint: Hint) -> bool:
        now = time.time()

        # global rate limit
        if (now - self._last_global_t) < self.min_interval:
            return False

        # per-issue cooldown
        last = self._last_issue_t.get(hint.issue_code, 0.0)
        if (now - last) < self.cooldown:
            return False

        # confidence gate (minimal)
        if hint.confidence < 0.25:
            return False

        self._last_global_t = now
        self._last_issue_t[hint.issue_code] = now
        return True

    def emit(self, hint: Hint) -> None:
        if self.enable_voice and self._tts is not None:
            self._tts.speak(hint.message)
