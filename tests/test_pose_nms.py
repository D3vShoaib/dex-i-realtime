from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dexi.ecpose_adapter import suppress_duplicate_poses
from dexi.types import Detection


def _detection(score: float, offset: float = 0,
               different_pose: bool = False) -> Detection:
    keypoints = np.array([
        [50, 20], [47, 18], [53, 18], [44, 20], [56, 20],
        [40, 45], [60, 45], [35, 75], [65, 75], [30, 100], [70, 100],
        [43, 100], [57, 100], [42, 140], [58, 140], [41, 180], [59, 180],
    ], dtype=np.float32)
    if different_pose:
        keypoints[:, 0] = 100 - keypoints[:, 0]
        keypoints[7:, 1] -= 35
    keypoints[:, :2] += offset
    keypoints = np.column_stack([keypoints, np.ones(17, np.float32)])
    xy = keypoints[:, :2]
    box = np.array([xy[:, 0].min() - 5, xy[:, 1].min() - 5,
                    xy[:, 0].max() + 5, xy[:, 1].max() + 5], np.float32)
    return Detection(box, score, keypoints, 0, 0.0)


class PoseNmsTests(unittest.TestCase):
    def test_removes_lower_confidence_duplicate(self):
        strong = _detection(0.9)
        duplicate = _detection(0.6, offset=1)
        self.assertEqual(suppress_duplicate_poses([duplicate, strong]),
                         [strong])

    def test_keeps_overlapping_different_pose(self):
        first = _detection(0.9)
        second = _detection(0.8, different_pose=True)
        self.assertEqual(len(suppress_duplicate_poses([first, second])), 2)


if __name__ == '__main__':
    unittest.main()
