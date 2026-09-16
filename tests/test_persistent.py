from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dexi.persistent import (
    PersistentIDManager,
    _spatial_similarity,
    _upper_torso_box,
)
from dexi.types import Track


class _FakeEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, frame, boxes):
        return [np.array([1.0, 0.0, 0.0], np.float32) for _ in boxes]

    def embed_crop(self, crop):
        return np.array([1.0, 0.0, 0.0], np.float32)


class _PositionEncoder(_FakeEncoder):
    def __call__(self, frame, boxes):
        features = []
        for box in boxes:
            center_x = (box[0] + box[2]) * 0.5
            features.append(np.array([1.0, 0.0], np.float32)
                            if center_x < 100
                            else np.array([0.0, 1.0], np.float32))
        return features


class PersistentIDManagerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((200, 200, 3), np.uint8)
        self.box = np.array([50, 20, 100, 180], np.float32)

    def _track(self, tracker_id: int, frame_id: int, shift: float = 0) -> Track:
        offset = np.array([shift, 0, shift, 0], np.float32)
        return Track(tracker_id, self.box + offset, 0.9, None,
                     frame_id, frame_id / 5)

    @patch('dexi.persistent.OSNetEncoder', _FakeEncoder)
    def test_recovers_id_after_long_occlusion_and_bounds_gallery(self):
        manager = PersistentIDManager(lost_ttl=300, gallery_size=3,
                                      refresh_interval=25)

        persistent_id = manager.update(
            [self._track(10, 0)], self.frame, 0
        )[0].track_id
        for frame_id in (25, 50, 75, 100):
            current_id = manager.update(
                [self._track(10, frame_id)], self.frame, frame_id
            )[0].track_id
            self.assertEqual(current_id, persistent_id)

        self.assertEqual(len(manager.bank[persistent_id]['gallery']), 3)
        manager.update([], self.frame, 101)
        recovered_id = manager.update(
            [self._track(99, 250, shift=2)], self.frame, 250
        )[0].track_id
        self.assertEqual(recovered_id, persistent_id)

    def test_spatial_similarity_rewards_same_location(self):
        nearby = self.box + np.array([5, 0, 5, 0], np.float32)
        distant = self.box + np.array([500, 0, 500, 0], np.float32)
        self.assertGreater(_spatial_similarity(self.box, nearby),
                           _spatial_similarity(self.box, distant))
        self.assertAlmostEqual(_spatial_similarity(self.box, self.box), 1.0)

    def test_upper_torso_crop_uses_shoulders_and_hips(self):
        keypoints = np.zeros((17, 3), np.float32)
        keypoints[[5, 6, 11, 12]] = [
            [60, 60, 1], [90, 60, 1], [65, 120, 1], [85, 120, 1],
        ]
        crop = _upper_torso_box(self.box, keypoints)
        self.assertTrue(np.allclose(crop, [54, 54, 96, 126]))

    @patch('dexi.persistent.OSNetEncoder', _PositionEncoder)
    def test_new_track_can_reclaim_id_from_wrong_live_track(self):
        manager = PersistentIDManager(refresh_interval=1000)
        left = np.array([10, 20, 60, 180], np.float32)
        right = np.array([120, 20, 170, 180], np.float32)
        initial = [
            Track(1, left, 0.9, None, 0, 0),
            Track(2, right, 0.9, None, 0, 0),
        ]
        manager.update(initial, self.frame, 0)

        # Raw tracker 1 followed the right-hand person through an overlap;
        # the left-hand person then reappeared under a new raw tracker ID.
        crossed = [
            Track(1, right, 0.9, None, 1, 0.2),
            Track(3, left, 0.9, None, 1, 0.2),
        ]
        corrected = manager.update(crossed, self.frame, 1)
        by_tid = {track.tid: track.track_id for track in corrected}
        self.assertEqual(by_tid, {1: 2, 3: 1})

    @patch('dexi.persistent.OSNetEncoder', _PositionEncoder)
    def test_two_live_tracks_cannot_share_a_persistent_id(self):
        manager = PersistentIDManager(refresh_interval=1000)
        left = np.array([10, 20, 60, 180], np.float32)
        right = np.array([120, 20, 170, 180], np.float32)
        manager.update([Track(1, left, 0.9, None, 0, 0)], self.frame, 0)
        manager.tid_to_pid[2] = 1

        tracks = [
            Track(1, right, 0.9, None, 1, 0.2),
            Track(2, left, 0.9, None, 1, 0.2),
        ]
        updated = manager.update(tracks, self.frame, 1)
        persistent_ids = [track.track_id for track in updated]
        self.assertEqual(len(persistent_ids), len(set(persistent_ids)))


if __name__ == '__main__':
    unittest.main()
