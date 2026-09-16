"""Persistent identities through global appearance, pose, and location matching.

TrackTrack supplies short-term motion tracks. At a fixed interval, and whenever
tracks overlap or a new tracker ID appears, every visible track is jointly
matched against every retained identity with a one-to-one Hungarian assignment.
ReID uses pose-derived upper-torso crops to reduce background and bystander
contamination.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .pose_rerank import oks
from .reid import IR_PATH as REID_IR
from .reid import AppearanceCache, OSNetEncoder
from .types import Track


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / max(1.0, a_area + b_area - inter))


def _spatial_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Soft center-distance score normalized by person size."""
    ac = (np.asarray(a[:2]) + np.asarray(a[2:])) * 0.5
    bc = (np.asarray(b[:2]) + np.asarray(b[2:])) * 0.5
    a_diag = float(np.linalg.norm(np.asarray(a[2:]) - np.asarray(a[:2])))
    b_diag = float(np.linalg.norm(np.asarray(b[2:]) - np.asarray(b[:2])))
    scale = max(1.0, 0.5 * (a_diag + b_diag))
    return float(np.exp(-float(np.linalg.norm(ac - bc)) / scale))


def _upper_torso_box(box: np.ndarray, keypoints: np.ndarray | None) -> np.ndarray:
    """Return a padded shoulder-to-hip crop from COCO pose keypoints."""
    box = np.asarray(box, dtype=np.float32).reshape(4)
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    if keypoints is not None:
        kp = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
        torso = kp[[5, 6, 11, 12]]  # left/right shoulders and hips
        visible = torso[:, 2] > 0
        if visible.sum() >= 3:
            points = torso[visible, :2]
            tx1, ty1 = points.min(axis=0)
            tx2, ty2 = points.max(axis=0)
            tw, th = max(1.0, tx2 - tx1), max(1.0, ty2 - ty1)
            return np.array([tx1 - 0.20 * tw, ty1 - 0.10 * th,
                             tx2 + 0.20 * tw, ty2 + 0.10 * th],
                            dtype=np.float32)

    return np.array([x1 + 0.10 * bw, y1 + 0.15 * bh,
                     x2 - 0.10 * bw, y1 + 0.60 * bh], dtype=np.float32)


class PersistentIDManager:
    def __init__(self, reid_model: str = REID_IR,
                 rebind_thr: float = 0.55, reid_w: float = 0.80,
                 spatial_w: float = 0.10, overlap_thr: float = 0.5,
                 lost_ttl: int = 300, gallery_size: int = 5,
                 refresh_interval: int = 5, continuity_bonus: float = 0.0,
                 gallery_update_thr: float = 0.60,
                 enable: bool = True, num_threads: int | None = None):
        if reid_w < 0 or spatial_w < 0 or reid_w + spatial_w > 1:
            raise ValueError('reid_w and spatial_w must be non-negative and sum to at most 1')
        self.enable = enable
        self.rebind_thr = rebind_thr
        self.reid_w = reid_w
        self.spatial_w = spatial_w
        self.pose_w = 1.0 - reid_w - spatial_w
        self.overlap_thr = overlap_thr
        self.lost_ttl = max(1, lost_ttl)
        self.gallery_size = max(1, gallery_size)
        self.refresh_interval = max(1, refresh_interval)
        self.continuity_bonus = max(0.0, continuity_bonus)
        self.gallery_update_thr = gallery_update_thr
        self.encoder = OSNetEncoder(reid_model, num_threads=num_threads) if enable else None
        self.cache = AppearanceCache()
        self.tid_to_pid: dict[int, int] = {}
        self.bank: dict[int, dict] = {}
        self.next_pid = 1
        self.last_assignment_frame = -self.refresh_interval
        self.reid_calls = 0  # number of torso embeddings computed
        self.rebinds = 0
        self.frames = 0

    def _features(self, frame_bgr: np.ndarray,
                  tracks: list[Track]) -> list[np.ndarray | None]:
        if self.encoder is None or not tracks:
            return [None] * len(tracks)
        boxes = np.stack([
            _upper_torso_box(track.bbox_xyxy, track.keypoints)
            for track in tracks
        ]).astype(np.float32)
        features = self.encoder(frame_bgr, boxes)
        self.reid_calls += len(tracks)
        return features

    def _thumb(self, frame_bgr: np.ndarray, track: Track) -> np.ndarray | None:
        box = _upper_torso_box(track.bbox_xyxy, track.keypoints)
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return cv2.resize(frame_bgr[y1:y2, x1:x2], (64, 128))

    def _remember_feature(self, pid: int, feat: np.ndarray | None,
                          score: float) -> np.ndarray | None:
        ema = self.cache.update(pid, feat, score)
        if feat is None:
            return ema
        normalized = np.asarray(feat, dtype=np.float32)
        normalized /= float(np.linalg.norm(normalized) + 1e-9)
        gallery = self.bank[pid].setdefault('gallery', [])
        gallery.append(normalized)
        del gallery[:-self.gallery_size]
        return ema

    @staticmethod
    def _appearance_similarity(feat: np.ndarray, rec: dict) -> float:
        references = list(rec.get('gallery', []))
        if rec.get('feat') is not None:
            references.append(rec['feat'])
        if not references:
            return -1.0
        return max(float(feat @ reference) for reference in references)

    def _ambiguous_tids(self, tracks: list[Track]) -> set[int]:
        ambiguous: set[int] = set()
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                if _iou(tracks[i].bbox_xyxy, tracks[j].bbox_xyxy) > self.overlap_thr:
                    ambiguous.add(tracks[i].track_id)
                    ambiguous.add(tracks[j].track_id)
        return ambiguous

    def _pair_score(self, track: Track, feat: np.ndarray | None,
                    rec: dict) -> float:
        if feat is None:
            return -1.0
        appearance = self._appearance_similarity(feat, rec)
        pose_score = 0.0
        if track.keypoints is not None and rec.get('pose') is not None:
            pose_score = oks(rec['pose'], track.keypoints, track.bbox_xyxy)
        location = _spatial_similarity(rec['bbox'], track.bbox_xyxy)
        return (self.reid_w * appearance + self.pose_w * pose_score
                + self.spatial_w * location)

    def _new_identity(self, track: Track, feat: np.ndarray | None,
                      frame_bgr: np.ndarray, frame_id: int) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.bank[pid] = {'gallery': []}
        ema = self._remember_feature(pid, feat, track.score)
        self.bank[pid].update(feat=ema, pose=track.keypoints,
                              bbox=track.bbox_xyxy.copy(),
                              last_frame=frame_id,
                              thumb=self._thumb(frame_bgr, track))
        return pid

    def _global_assignment(self, tracks: list[Track], frame_bgr: np.ndarray,
                           frame_id: int, ambiguous: set[int]) -> None:
        features = self._features(frame_bgr, tracks)
        candidate_pids = [
            pid for pid, rec in self.bank.items()
            if frame_id - rec['last_frame'] <= self.lost_ttl
            and rec.get('feat') is not None
        ]
        old_mapping = dict(self.tid_to_pid)
        created_pids: set[int] = set()

        if not candidate_pids:
            assignments = [self._new_identity(t, f, frame_bgr, frame_id)
                           for t, f in zip(tracks, features)]
            created_pids.update(assignments)
        else:
            n_tracks, n_ids = len(tracks), len(candidate_pids)
            base = np.full((n_tracks, n_ids), -1.0, dtype=np.float32)
            objective = np.full((n_tracks, n_ids + n_tracks),
                                self.rebind_thr, dtype=np.float32)
            for row, (track, feat) in enumerate(zip(tracks, features)):
                for col, pid in enumerate(candidate_pids):
                    score = self._pair_score(track, feat, self.bank[pid])
                    base[row, col] = score
                    objective[row, col] = score
                    if old_mapping.get(track.track_id) == pid:
                        objective[row, col] += self.continuity_bonus

            rows, cols = linear_sum_assignment(objective, maximize=True)
            chosen = dict(zip(rows.tolist(), cols.tolist()))
            assignments = []
            for row, (track, feat) in enumerate(zip(tracks, features)):
                col = chosen[row]
                if col < n_ids and base[row, col] >= self.rebind_thr:
                    assignments.append(candidate_pids[col])
                else:
                    pid = self._new_identity(track, feat, frame_bgr, frame_id)
                    assignments.append(pid)
                    created_pids.add(pid)

        self.tid_to_pid = {}
        for track, feat, pid in zip(tracks, features, assignments):
            tid = track.track_id
            previous = old_mapping.get(tid)
            if previous is not None and previous != pid:
                self.rebinds += 1
            elif previous is None and pid in candidate_pids:
                self.rebinds += 1
            self.tid_to_pid[tid] = pid
            rec = self.bank[pid]

            # Never learn from overlapping crops. Otherwise only accept a new
            # view when it is reasonably consistent with this identity.
            stable_mapping = previous == pid
            if (pid not in created_pids and stable_mapping
                    and tid not in ambiguous and feat is not None):
                similarity = self._appearance_similarity(feat, rec)
                if similarity >= self.gallery_update_thr:
                    rec['feat'] = self._remember_feature(pid, feat, track.score)
                    rec['thumb'] = self._thumb(frame_bgr, track)
            rec['last_frame'] = frame_id
            rec['bbox'] = track.bbox_xyxy.copy()
            if track.keypoints is not None:
                rec['pose'] = track.keypoints
            track.tid, track.track_id = tid, pid
            track.reid_feat = rec.get('feat')

        self.last_assignment_frame = frame_id

    def _continue_assignments(self, tracks: list[Track], frame_bgr: np.ndarray,
                              frame_id: int, ambiguous: set[int]) -> None:
        for track in tracks:
            tid = track.track_id
            pid = self.tid_to_pid[tid]
            rec = self.bank[pid]
            rec['last_frame'] = frame_id
            rec['bbox'] = track.bbox_xyxy.copy()
            if track.keypoints is not None:
                rec['pose'] = track.keypoints
            if tid not in ambiguous:
                rec['thumb'] = self._thumb(frame_bgr, track)
            track.tid, track.track_id = tid, pid
            track.reid_feat = rec.get('feat')

    def update(self, tracks: list[Track], frame_bgr: np.ndarray,
               frame_id: int) -> list[Track]:
        if not self.enable:
            return tracks
        self.frames += 1
        ambiguous = self._ambiguous_tids(tracks)
        raw_tids = {track.track_id for track in tracks}
        mapped_pids = [self.tid_to_pid[tid] for tid in raw_tids
                       if tid in self.tid_to_pid]
        has_new_track = any(tid not in self.tid_to_pid for tid in raw_tids)
        has_duplicate_id = len(mapped_pids) != len(set(mapped_pids))
        interval_due = frame_id - self.last_assignment_frame >= self.refresh_interval
        needs_assignment = (bool(tracks) and
                            (not self.bank or has_new_track or has_duplicate_id
                             or bool(ambiguous) or interval_due))

        if needs_assignment:
            self._global_assignment(tracks, frame_bgr, frame_id, ambiguous)
        elif tracks:
            self._continue_assignments(tracks, frame_bgr, frame_id, ambiguous)

        live_pids = {track.track_id for track in tracks}
        dead = [pid for pid, rec in self.bank.items()
                if frame_id - rec['last_frame'] > self.lost_ttl]
        for pid in dead:
            del self.bank[pid]
        self.cache.drop(set(self.bank))

        # Raw tracker IDs are short-lived routing keys. Retaining only currently
        # visible mappings prevents a recycled tracker ID from inheriting a PID.
        self.tid_to_pid = {
            track.tid: track.track_id for track in tracks
            if track.track_id in live_pids
        }
        return tracks
