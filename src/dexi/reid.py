"""Lightweight ReID: OSNet-x0.25 MSMT17 FP32 OpenVINO IR on CPU.

Persistent identity policy and the bounded appearance gallery live in
persistent.py. This module only owns embedding preprocessing/inference and EMA.
"""
from __future__ import annotations

import cv2
import numpy as np
import openvino as ov

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
IR_PATH = 'models/osnet_x0_25_msmt17.xml'


class AppearanceCache:
    """One EMA vector per track_id; avoids recompute on stable tracks."""

    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha
        self.bank: dict[int, np.ndarray] = {}

    def update(self, track_id: int, feat: np.ndarray | None, score: float) -> np.ndarray | None:
        if feat is None:
            return self.bank.get(track_id)
        feat = np.asarray(feat, dtype=np.float32)
        n = float(np.linalg.norm(feat) + 1e-9)
        feat = feat / n
        beta = self.alpha + (1 - self.alpha) * (1 - float(score))
        prev = self.bank.get(track_id)
        ema = feat if prev is None else beta * prev + (1 - beta) * feat
        ema = ema / (float(np.linalg.norm(ema) + 1e-9))
        self.bank[track_id] = ema.astype(np.float32)
        return self.bank[track_id]

    def get(self, track_id: int) -> np.ndarray | None:
        return self.bank.get(track_id)

    def drop(self, keep: set[int]) -> None:
        for k in list(self.bank):
            if k not in keep:
                del self.bank[k]


class OSNetEncoder:
    """OSNet-x0.25 MSMT17 embedding via the FP32 OpenVINO IR (CPU).

    Preprocess matches torchreid: resize to 256x128 (HxW), BGR->RGB,
    normalize mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225].
    IR source: models/osnet_x0_25_msmt17.onnx (torchreid MSMT17 weights) -> OpenVINO.
    """

    def __init__(self, ir_path: str = IR_PATH, device: str = 'CPU',
                 num_threads: int | None = None):
        self.ir_path = ir_path
        props = {} if num_threads is None else {ov.properties.inference_num_threads: num_threads}
        self.model = ov.Core().compile_model(ir_path, device, props)
        self.inp = self.model.input(0)
        self.out = self.model.output(0)

    @staticmethod
    def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 HxWx3 -> NCHW float32 (1,3,256,128): RGB + ImageNet norm."""
        crop = cv2.resize(crop_bgr, (128, 256))
        arr = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(((arr - _MEAN) / _STD).transpose(2, 0, 1)[None])

    def embed_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Embed an already-cropped BGR patch (e.g. cached thumb)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        f = self.model({self.inp: self._preprocess(crop_bgr)})[self.out][0]
        return (f / (np.linalg.norm(f) + 1e-9)).astype(np.float32)

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray):
        if len(boxes_xyxy) == 0:
            return []
        h, w = frame_bgr.shape[:2]
        feats = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                feats.append(None)
                continue
            feats.append(self.embed_crop(frame_bgr[y1:y2, x1:x2]))
        return feats
