"""Pipeline: FrameSource -> ECPose (OpenVINO) -> TrackTrack -> ReID gate + OKS -> tracks.

One inference implementation: ECPose-M and OSNet both run as FP32 OpenVINO IR on CPU.
"""
from __future__ import annotations

from .ecpose_adapter import IR_PATH as ECPOSE_IR
from .ecpose_adapter import ECPoseDetector
from .frame_source import FrameSource
from .persistent import PersistentIDManager
from .reid import IR_PATH as REID_IR
from .track_adapter import TrackAdapter
from .types import Track


class DexiPipeline:
    def __init__(self, det_thresh: float = 0.4,
                 tracker_cfg: str = 'configs/tracktrack_dexi.yaml',
                 ecpose_ir: str = ECPOSE_IR, reid: bool = True,
                 reid_ir: str = REID_IR, rebind_thr: float = 0.55,
                 lost_ttl: int = 20, num_threads: int | None = None):
        self.detector = ECPoseDetector(ir_path=ecpose_ir, thresh=det_thresh,
                                       num_threads=num_threads)
        self.tracker = TrackAdapter(cfg_path=tracker_cfg)
        self.manager = PersistentIDManager(reid_model=reid_ir, rebind_thr=rebind_thr,
                                           lost_ttl=lost_ttl, enable=reid,
                                           num_threads=num_threads)

    def process_frame(self, frame_bgr, timestamp: float, frame_id: int) -> tuple[list, list[Track]]:
        dets = self.detector.infer(frame_bgr, timestamp, frame_id)
        tracks = self.tracker.update(dets, frame_bgr, frame_id, timestamp)
        tracks = self.manager.update(tracks, frame_bgr, frame_id)
        return dets, tracks

    def run(self, source: FrameSource):
        for frame, ts, fid in source:
            dets, tracks = self.process_frame(frame, ts, fid)
            yield frame, dets, tracks, ts, fid
