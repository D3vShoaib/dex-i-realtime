"""Final persistent ID layer: ambiguity gate -> OSNet ReID -> OKS rerank -> ID map.

OSNet ReID runs the FP32 OpenVINO IR on CPU (models/osnet_x0_25_msmt17.xml/.bin).

TrackTrack owns motion/IoU association and lifecycle. This manager only steps in
for difficult cases, keeping ReID strictly selective:

  * a never-seen tracker id appears  -> lost-track recovery candidate
    (compute its crop feat ONCE, compare vs bank of recently lost pids)
  * two live tracks overlap heavily  -> compute feats for the pair, fuse with OKS
  * stable, non-overlapping tracks    -> NO ReID call at all

Rebind score (pose never dominates):  fused = 0.75 * cos_sim + 0.25 * oks.
A rebind needs fused >= rebind_thr (default 0.55).
"""
from __future__ import annotations

import cv2
import numpy as np

from .pose_rerank import oks
from .reid import IR_PATH as REID_IR
from .reid import AppearanceCache, OSNetEncoder, should_use_reid
from .types import Track


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(1.0, (a[2] - a[0]) * (a[3] - a[1])) + max(1.0, (b[2] - b[0]) * (b[3] - b[1])) - inter
    return float(inter / max(1.0, ua))


class PersistentIDManager:
    def __init__(self, reid_model: str = REID_IR,
                 rebind_thr: float = 0.55, reid_w: float = 0.75,
                 overlap_thr: float = 0.5, lost_ttl: int = 20,
                 enable: bool = True, num_threads: int | None = None):
        self.enable = enable
        self.rebind_thr = rebind_thr
        self.reid_w = reid_w
        self.overlap_thr = overlap_thr
        self.lost_ttl = lost_ttl
        self.encoder = OSNetEncoder(reid_model, num_threads=num_threads) if enable else None
        self.cache = AppearanceCache()
        self.tid_to_pid: dict[int, int] = {}
        self.bank: dict[int, dict] = {}  # pid -> {feat, pose, bbox, last_frame}
        self.next_pid = 1
        # stats
        self.reid_calls = 0
        self.rebinds = 0
        self.frames = 0

    def _feat(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        if self.encoder is None:
            return None
        self.reid_calls += 1
        feats = self.encoder(frame_bgr, np.asarray(box, np.float32).reshape(-1, 4))
        return feats[0] if feats else None

    def _thumb(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        try:
            return cv2.resize(frame_bgr[y1:y2, x1:x2], (64, 128))
        except Exception:
            return None

    def update(self, tracks: list[Track], frame_bgr: np.ndarray, frame_id: int) -> list[Track]:
        if not self.enable:
            return tracks
        self.frames += 1
        live_tids = {t.track_id for t in tracks}

        # 1. overlap detection among live tracks -> ambiguous set
        ambiguous: set[int] = set()
        boxes = [t.bbox_xyxy for t in tracks]
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                if _iou(boxes[i], boxes[j]) > self.overlap_thr:
                    ambiguous.add(tracks[i].track_id)
                    ambiguous.add(tracks[j].track_id)

        # 2. resolve each tracker id to a persistent id
        for t in tracks:
            tid = t.track_id
            if tid in self.tid_to_pid:
                pid = self.tid_to_pid[tid]
                # stable track: no ReID unless overlapping someone
                if tid in ambiguous and should_use_reid(0.0, 1.0, 0):
                    f = self._feat(frame_bgr, t.bbox_xyxy)
                    self.cache.update(pid, f, t.score)
                    self.bank[pid].update(feat=self.cache.get(pid), pose=t.keypoints,
                                          bbox=t.bbox_xyxy.copy(), last_frame=frame_id)
                else:
                    self.bank[pid]['last_frame'] = frame_id
                    if t.keypoints is not None:
                        self.bank[pid]['pose'] = t.keypoints
                    self.bank[pid]['bbox'] = t.bbox_xyxy.copy()
                self.bank[pid]['thumb'] = self._thumb(frame_bgr, t.bbox_xyxy)
                t.tid, t.track_id = tid, pid
                t.reid_feat = self.cache.get(pid)
                continue

            # 3. new tracker id -> recovery candidate (gate: lost reactivation case)
            f = self._feat(frame_bgr, t.bbox_xyxy) if should_use_reid(0.0, 0.0, 1) else None
            best_pid, best_fused = -1, -1.0
            if f is not None:
                for pid, rec in self.bank.items():
                    if rec['last_frame'] >= frame_id:  # still live under another tid
                        continue
                    if frame_id - rec['last_frame'] > self.lost_ttl:
                        continue
                    bf = rec.get('feat')
                    if bf is None:
                        continue
                    cos = float(f @ bf)
                    pose_score = 0.0
                    if t.keypoints is not None and rec.get('pose') is not None:
                        pose_score = oks(rec['pose'], t.keypoints, t.bbox_xyxy)
                    fused = self.reid_w * cos + (1 - self.reid_w) * pose_score
                    if fused > best_fused:
                        best_fused, best_pid = fused, pid
            if best_pid > 0 and best_fused >= self.rebind_thr:
                pid = best_pid
                self.rebinds += 1
            else:
                pid = self.next_pid
                self.next_pid += 1
            self.tid_to_pid[tid] = pid
            ema = self.cache.update(pid, f, t.score)
            self.bank[pid] = {'feat': ema, 'pose': t.keypoints,
                              'bbox': t.bbox_xyxy.copy(), 'last_frame': frame_id,
                              'thumb': self._thumb(frame_bgr, t.bbox_xyxy)}
            t.tid, t.track_id = tid, pid
            t.reid_feat = ema

        # 4. refresh appearance of just-lost pids from their last thumb, so a
        # later reappearance compares against how they looked when last seen
        # (1 encoder call per loss event; stable live tracks still cost nothing)
        live_pids = {t.track_id for t in tracks}
        if self.encoder is not None:
            for pid, rec in self.bank.items():
                if pid in live_pids or rec['last_frame'] != frame_id - 1:
                    continue
                thumb = rec.get('thumb')
                if thumb is None:
                    continue
                f = self.encoder.embed_crop(thumb)
                if f is not None:
                    self.reid_calls += 1
                    rec['feat'] = self.cache.update(pid, f, 0.5)

        # 5. expire long-lost pids
        dead = [p for p, r in self.bank.items() if frame_id - r['last_frame'] > self.lost_ttl]
        for p in dead:
            del self.bank[p]
        self.cache.drop(set(self.bank))
        # release tids of tracks gone from the tracker so a recycled tid can't collide
        for tid in list(self.tid_to_pid):
            if tid not in live_tids:
                # keep mapping while its pid is still in bank (allows rebind); drop otherwise
                if self.tid_to_pid[tid] not in self.bank:
                    del self.tid_to_pid[tid]
        return tracks
