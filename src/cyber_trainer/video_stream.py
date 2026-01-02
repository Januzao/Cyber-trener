from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import requests


@dataclass
class FramePacket:
    frame: np.ndarray
    ts_ms: int  # local receive timestamp (ms)


class MjpegStream:
    """
    Very simple MJPEG reader over HTTP.
    It parses JPEG frames by scanning for 0xFFD8 ... 0xFFD9 markers.

    Works well with many phone "IP webcam" apps that expose /video endpoint.
    """
    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self._resp: Optional[requests.Response] = None
        self._buf = bytearray()

    def connect(self) -> None:
        self.close()
        self._resp = requests.get(self.url, stream=True, timeout=(self.timeout_s, self.timeout_s))
        self._resp.raise_for_status()

    def close(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
        self._resp = None
        self._buf = bytearray()

    def read_frame(self) -> Optional[FramePacket]:
        if self._resp is None:
            self.connect()

        assert self._resp is not None
        for chunk in self._resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            self._buf.extend(chunk)

            a = self._buf.find(b"\xff\xd8")  # SOI
            b = self._buf.find(b"\xff\xd9")  # EOI
            if a != -1 and b != -1 and b > a:
                jpg = bytes(self._buf[a : b + 2])
                del self._buf[: b + 2]

                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                ts_ms = int(time.time() * 1000)
                return FramePacket(frame=frame, ts_ms=ts_ms)

        return None


class LatestFrameGrabber(threading.Thread):
    """
    Background thread that always keeps only the latest decoded frame.
    This is the simplest backpressure strategy: drop old frames, keep latency low.
    """
    def __init__(self, name: str, url: str, reconnect_sleep_s: float = 1.0) -> None:
        super().__init__(daemon=True, name=name)
        self.stream = MjpegStream(url)
        self.reconnect_sleep_s = reconnect_sleep_s

        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                pkt = self.stream.read_frame()
                if pkt is None:
                    continue
                with self._lock:
                    self._latest = pkt
            except Exception:
                # reconnect loop
                try:
                    self.stream.close()
                except Exception:
                    pass
                time.sleep(self.reconnect_sleep_s)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.stream.close()
        except Exception:
            pass

    def get_latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest
