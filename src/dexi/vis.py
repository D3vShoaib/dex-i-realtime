"""Visualization: bbox + track id + 17-kpt skeleton (COCO, 0-based)."""
from __future__ import annotations

import cv2
import numpy as np

# 1-based COCO skeleton -> 0-based
_SKELETON = [(16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
             (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
             (2, 4), (3, 5), (4, 6), (5, 7)]
SKELETON = [(a - 1, b - 1) for a, b in _SKELETON]

PALETTE = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255),
           (0, 255, 255), (128, 255, 0), (255, 255, 0), (0, 128, 255)]


def draw_tracks(frame_bgr: np.ndarray, tracks, draw_skeleton: bool = True) -> np.ndarray:
    out = frame_bgr
    for t in tracks:
        c = PALETTE[t.track_id % len(PALETTE)]
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, f'ID {t.track_id} {t.score:.2f}', (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        if draw_skeleton and t.keypoints is not None:
            kp = np.asarray(t.keypoints).reshape(17, 3)
            for a, b in SKELETON:
                if kp[a, 2] > 0 and kp[b, 2] > 0:
                    cv2.line(out, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[b, 0]), int(kp[b, 1])), (255, 128, 0), 2)
            for x, y, v in kp:
                if v > 0:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    return out
