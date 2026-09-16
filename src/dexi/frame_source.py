"""Generic FrameSource: pipeline starts only after a decoded frame exists.

Yields (frame_bgr, timestamp, frame_id). Tracking logic never touches I/O.
Drop-stale rule for realtime sources: keep only the most recent frame.
"""
from __future__ import annotations

from typing import Iterator, Protocol, Tuple
import cv2
import numpy as np


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int]]: ...
    def close(self) -> None: ...


class Mp4Source:
    """File source for offline demo/validation. Sequential, no dropping needed."""

    def __init__(self, path: str, limit: int = 0, stride: int = 1):
        self.path = path
        self.limit = limit
        self.stride = max(1, stride)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video: {path}')
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        self.n = 0  # emitted count
        self._fid = -1  # source frame index

    def __iter__(self):
        while True:
            ret, frame = self.cap.read()
            self._fid += 1
            if not ret:
                break
            if self._fid % self.stride != 0:
                continue
            ts = self._fid / self.fps
            yield frame, ts, self._fid
            self.n += 1
            if self.limit and self.n >= self.limit:
                break

    def close(self):
        self.cap.release()


class LatestFrameBuffer:
    """Helper for realtime sources (USB/RTSP/NVR): keep only most recent frame.

    Usage: producer thread calls .push(frame, ts, fid); consumer calls .pop_latest().
    If detector falls behind, stale frames are discarded.
    """

    def __init__(self):
        import queue
        self._q: queue.Queue = queue.Queue(maxsize=1)

    def push(self, frame: np.ndarray, timestamp: float, frame_id: int) -> None:
        if self._q.full():
            try:
                self._q.get_nowait()
            except Exception:
                pass
        try:
            self._q.put_nowait((frame, timestamp, frame_id))
        except Exception:
            pass

    def pop_latest(self, timeout: float = 1.0):
        import queue
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
