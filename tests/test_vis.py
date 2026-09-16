from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dexi.vis import _chest_position


class VisualizationTests(unittest.TestCase):
    def test_id_position_uses_torso_keypoints(self):
        keypoints = np.zeros((17, 3), np.float32)
        keypoints[[5, 6, 11, 12]] = [
            [40, 40, 1], [60, 40, 1], [45, 100, 1], [55, 100, 1],
        ]
        self.assertEqual(_chest_position(keypoints, [0, 0, 100, 200]),
                         (50, 61))

    def test_id_position_falls_back_to_upper_bbox(self):
        self.assertEqual(_chest_position(None, [0, 0, 100, 200]),
                         (50, 70))


if __name__ == '__main__':
    unittest.main()
