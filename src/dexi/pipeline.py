"""Pipeline: FrameSource -> ECPose -> TrackTrack -> (Phase 3: ReID gate + OKS) -> tracks."""
from __future__ import annotations

from .ecpose_adapter import ECPoseDetector
from .frame_source import FrameSource
from .persistent import PersistentIDManager
from .track_adapter import TrackAdapter
from .types import Track


class DexiPipeline:
    def __init__(self, weights: str = 'ecpose_m_o3652coco.pth', device: str = 'cpu',
                 det_thresh: float = 0.4, tracker_cfg: str = 'configs/tracktrack_dexi.yaml',
                 backend: str = 'torch', onnx_path: str = 'ecpose_m_o3652coco.onnx',
                 intra_threads: int = 4, reid_model: str | None = None,
                 rebind_thr: float = 0.55, lost_ttl: int = 20):
        self.detector = ECPoseDetector(weights=weights, device=device, thresh=det_thresh,
                                       backend=backend, onnx_path=onnx_path,
                                       intra_threads=intra_threads)
        self.tracker = TrackAdapter(cfg_path=tracker_cfg)
        self.manager = PersistentIDManager(reid_model=reid_model or 'osnet_x0_25_msmt17.onnx',
                                           rebind_thr=rebind_thr, lost_ttl=lost_ttl,
                                           enable=reid_model is not None)

    def process_frame(self, frame_bgr, timestamp: float, frame_id: int) -> tuple[list, list[Track]]:
        dets = self.detector.infer(frame_bgr, timestamp, frame_id)
        tracks = self.tracker.update(dets, frame_bgr, frame_id, timestamp)
        tracks = self.manager.update(tracks, frame_bgr, frame_id)
        return dets, tracks

    def run(self, source: FrameSource):
        for frame, ts, fid in source:
            dets, tracks = self.process_frame(frame, ts, fid)
            yield frame, dets, tracks, ts, fid
