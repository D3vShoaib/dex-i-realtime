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
SKELETON_OPACITY = 0.4


def _chest_position(keypoints, bbox_xyxy) -> tuple[int, int]:
    """Place an ID between the shoulder and hip centers."""
    if keypoints is not None:
        kp = np.asarray(keypoints).reshape(17, 3)
        torso = kp[[5, 6, 11, 12]]
        if np.all(torso[:, 2] > 0):
            shoulders = (kp[5, :2] + kp[6, :2]) * 0.5
            hips = (kp[11, :2] + kp[12, :2]) * 0.5
            chest = shoulders * 0.65 + hips * 0.35
            return int(chest[0]), int(chest[1])

    x1, y1, x2, y2 = np.asarray(bbox_xyxy).reshape(4)
    return int((x1 + x2) * 0.5), int(y1 + (y2 - y1) * 0.35)


def draw_tracks(frame_bgr: np.ndarray, tracks, draw_skeleton: bool = True) -> np.ndarray:
    out = frame_bgr
    skeleton_overlay = out.copy()
    labels = []

    for t in tracks:
        labels.append((str(t.track_id),
                       _chest_position(t.keypoints, t.bbox_xyxy)))
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

    for text, center in labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.7, 2
        (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
        position = (center[0] - width // 2, center[1] + height // 2)
        # A thin dark outline keeps the number readable without a badge that
        # would hide clothing details from downstream multimodal models.
        cv2.putText(out, text, position, font, scale, (0, 0, 0), 5,
                    cv2.LINE_AA)
        cv2.putText(out, text, position, font, scale, PARROT_GREEN, thickness,
                    cv2.LINE_AA)

    return out
