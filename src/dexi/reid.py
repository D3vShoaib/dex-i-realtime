"""Selective lightweight ReID (Phase 3 hook, motion-only in Phase 1/2).

Target: OSNet-x0.25 MSMT17 -> ONNX -> ONNX Runtime on Ryzen 3 5300U.
Policy: call ReID ONLY for difficult cases; cache one EMA vector per track.

  should_use_reid(...) -> True when:
    - top-2 motion costs are close (ambiguous), or
    - max IoU overlap with another det/track > overlap_thr, or
    - reactivating a lost track (lost_age > 0)

Phase 1: gate is implemented + EMA cache works; actual encoder returns None
(no model yet), so tracker runs motion-only. Phase 3 fills OSNetEncoder.
"""
from __future__ import annotations

import numpy as np


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


def should_use_reid(cost_margin: float, max_overlap: float, lost_age: int,
                    margin_thr: float = 0.1, overlap_thr: float = 0.5) -> bool:
    if lost_age > 0:
        return True
    if max_overlap > overlap_thr:
        return True
    if cost_margin < margin_thr:
        return True
    return False


class OSNetEncoder:
    """Phase 3: loads osnet_x0_25_msmt17.onnx via ONNX Runtime.

    Preprocess matches torchreid: resize to 256x128 (HxW), BGR->RGB,
    normalize mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225].
    Currently raises until the ONNX file is exported (see tools/export_osnet_onnx.py).
    """

    def __init__(self, onnx_path: str = 'osnet_x0_25_msmt17.onnx'):
        self.onnx_path = onnx_path
        self.session = None

    def load(self):
        import onnxruntime as ort
        self.session = ort.InferenceSession(self.onnx_path,
                                            providers=['CPUExecutionProvider'])
        return self

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        crop = cv2.resize(crop_bgr, (128, 256))
        arr = crop[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        return ((arr - mean) / std).transpose(2, 0, 1)[None]

    def embed_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Embed an already-cropped BGR patch (e.g. cached thumb)."""
        if self.session is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        inp = self.session.get_inputs()[0].name
        f = self.session.run(None, {inp: self._preprocess(crop_bgr)})[0][0]
        return (f / (np.linalg.norm(f) + 1e-9)).astype(np.float32)

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray):
        if self.session is None:
            return [None] * len(boxes_xyxy)
        import cv2
        feats = []
        inp = self.session.get_inputs()[0].name
        for b in boxes_xyxy:
            x1, y1, x2, y2 = [int(v) for v in b]
            h, w = frame_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                feats.append(None)
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))
            arr = crop[:, :, ::-1].astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], np.float32)
            std = np.array([0.229, 0.224, 0.225], np.float32)
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)[None]
            f = self.session.run(None, {inp: arr})[0][0]
            f = f / (np.linalg.norm(f) + 1e-9)
            feats.append(f.astype(np.float32))
        return feats
