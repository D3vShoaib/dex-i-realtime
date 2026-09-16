"""Shared detection/track types. Tracker-agnostic: only numpy, no torch/ultralytics."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    bbox_xyxy: np.ndarray  # (4,) float32, pixel coords in original frame
    score: float
    keypoints: np.ndarray  # (17, 3) float32: x, y (pixels), vis in {0,1}
    frame_id: int
    timestamp: float

    def __post_init__(self):
        self.bbox_xyxy = np.asarray(self.bbox_xyxy, dtype=np.float32).reshape(4)
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32).reshape(17, 3)


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    score: float
    keypoints: np.ndarray | None  # matched detection pose, else None
    frame_id: int
    timestamp: float
    lost: bool = False
    reid_feat: np.ndarray | None = field(default=None, repr=False)
    tid: int = -1  # raw tracker-side id before persistent remap (-1 = already persistent)
