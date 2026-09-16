"""Visualization: track id + 17-kpt skeleton (COCO, 0-based)."""
from __future__ import annotations

import cv2
import numpy as np

# 1-based COCO skeleton -> 0-based
_SKELETON = [(16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
             (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
             (2, 4), (3, 5), (4, 6), (5, 7)]
SKELETON = [(a - 1, b - 1) for a, b in _SKELETON]

PARROT_GREEN = (43, 173, 18)  # BGR for RGB #12AD2B
SKELETON_OPACITY = 0.6


def draw_tracks(frame_bgr: np.ndarray, tracks, draw_skeleton: bool = True) -> np.ndarray:
    out = frame_bgr
    skeleton_overlay = out.copy()
    labels = []

    for t in tracks:
        x1, y1 = [int(v) for v in t.bbox_xyxy[:2]]
        labels.append((f'ID {t.track_id}', (x1, max(0, y1 - 6))))
        if draw_skeleton and t.keypoints is not None:
            kp = np.asarray(t.keypoints).reshape(17, 3)
            for a, b in SKELETON:
                if kp[a, 2] > 0 and kp[b, 2] > 0:
                    cv2.line(skeleton_overlay, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[b, 0]), int(kp[b, 1])), PARROT_GREEN, 2)
            for x, y, v in kp:
                if v > 0:
                    cv2.circle(skeleton_overlay, (int(x), int(y)), 3, PARROT_GREEN, -1)

    if draw_skeleton:
        cv2.addWeighted(skeleton_overlay, SKELETON_OPACITY, out,
                        1.0 - SKELETON_OPACITY, 0, dst=out)

    for text, position in labels:
        cv2.putText(out, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, PARROT_GREEN, 2, cv2.LINE_AA)

    return out
