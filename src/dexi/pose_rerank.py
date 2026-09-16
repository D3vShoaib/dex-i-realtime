"""OKS / keypoint-distance rerank for ambiguous matches only.

Pose never dominates identity: caller applies this only when motion/IoU margin
is small, overlap is high, or a lost track is being reactivated.
Reference: MMPose OKS-based tracking docs; COCO 17-kpt sigmas.
"""
from __future__ import annotations

import numpy as np

# COCO keypoint sigmas (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
COCO_SIGMAS = np.array([.026, .025, .025, .035, .035, .035, .079, .079, .072, .072,
                        .062, .062, .107, .107, .087, .087, .089], dtype=np.float64)


def oks(track_kpts: np.ndarray, det_kpts: np.ndarray, det_bbox: np.ndarray) -> float:
    """Object Keypoint Similarity in [0,1]. Inputs: (17,3) x,y,vis; bbox xyxy."""
    t = np.asarray(track_kpts, dtype=np.float64).reshape(17, 3)
    d = np.asarray(det_kpts, dtype=np.float64).reshape(17, 3)
    vis = (t[:, 2] > 0) & (d[:, 2] > 0)
    if vis.sum() < 3:
        return 0.0
    area = max(1.0, float((det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])))
    s2 = area
    dx = t[vis, 0] - d[vis, 0]
    dy = t[vis, 1] - d[vis, 1]
    e = (dx ** 2 + dy ** 2) / (2 * s2 * (2 * COCO_SIGMAS[vis]) ** 2 + 1e-9)
    return float(np.exp(-e).mean())


def rerank(cost: np.ndarray, track_poses: list, det_poses: list,
           det_boxes: np.ndarray, w: float = 0.15,
           cost_margin: float = 0.1) -> np.ndarray:
    """Adjust motion/ReID cost with pose evidence on ambiguous rows only.

    A row is ambiguous if best and second-best costs differ by < cost_margin.
    cost is (n_tracks, n_dets); lower is better.
    """
    cost = np.asarray(cost, dtype=np.float64).copy()
    if cost.size == 0 or not track_poses or not det_poses:
        return cost
    nt, nd = cost.shape
    for i in range(nt):
        row = cost[i]
        order = np.argsort(row)
        if nd > 1 and not np.isfinite(row[order[1]]):
            continue
        ambiguous = nd == 1 or (row[order[1]] - row[order[0]] < cost_margin)
        if not ambiguous or track_poses[i] is None:
            continue
        for j in range(nd):
            if det_poses[j] is None:
                continue
            sim = oks(track_poses[i], det_poses[j], det_boxes[j])
            cost[i, j] = (1 - w) * cost[i, j] + w * (1.0 - sim)
    return cost
