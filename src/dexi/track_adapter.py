"""TrackTrack adapter: normalized Detections -> persistent tracks.

Wraps ultralytics TRACKTRACK so the tracker is independent of FrameSource.
Baseline runs motion-only (with_reid=False, gmc=none) for 5 FPS static video.
ReID (Phase 3) plugs into the same update() via encoder without changing callers.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import yaml

from .types import Detection, Track


class _Results:
    """Minimal Results-like stub satisfying TRACKTRACK.update + parse_bboxes."""
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


class TrackAdapter:
    def __init__(self, cfg_path: str = 'configs/tracktrack_dexi.yaml'):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.args = SimpleNamespace(**cfg)
        from ultralytics.trackers.track_tracker import TRACKTRACK
        self.tracker = TRACKTRACK(self.args)

    def update(self, dets: list[Detection], frame_bgr: np.ndarray,
               frame_id: int, timestamp: float) -> list[Track]:
        if dets:
            boxes = np.stack([d.bbox_xyxy for d in dets]).astype(np.float32)
            res = _Results(_xyxy_to_xywh(boxes),
                           np.array([d.score for d in dets], dtype=np.float32),
                           np.zeros(len(dets), dtype=np.float32))
        else:
            res = _Results(np.zeros((0, 4), np.float32),
                           np.zeros((0,), np.float32), np.zeros((0,), np.float32))
        # TRACKTRACK.update returns (N,8): x1,y1,x2,y2,id,score,cls,det_idx
        out = self.tracker.update(res, img=frame_bgr)
        tracks: list[Track] = []
        if out is None or len(out) == 0:
            return tracks
        for row in np.asarray(out):
            x1, y1, x2, y2, tid, score, _cls, didx = row.tolist()
            didx = int(didx)
            kp = dets[didx].keypoints if 0 <= didx < len(dets) else None
            tracks.append(Track(track_id=int(tid),
                                bbox_xyxy=np.array([x1, y1, x2, y2], np.float32),
                                score=float(score), keypoints=kp,
                                frame_id=frame_id, timestamp=timestamp))
        return tracks
