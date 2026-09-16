"""ECPose-M O365 adapter: Frame -> normalized Detection objects.

Upstream: EdgeCrafter ecpose (vendored under third_party/EdgeCrafter/ecpose).
Postprocessor in deploy mode returns (scores, labels, keypoints[17,2]) with NO
explicit bbox, so bbox_xyxy is derived from the keypoint envelope + padding.
"""
from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'third_party' / 'EdgeCrafter' / 'ecpose'))
from engine.core import YAMLConfig  # noqa: E402

from .types import Detection

PERSON_LABEL = 1
CONFIG = REPO / 'third_party' / 'EdgeCrafter' / 'ecpose' / 'configs' / 'ecpose' / 'ecpose_m_coco.yml'


class ECPoseDetector:
    def __init__(self, weights: str = 'ecpose_m_o3652coco.pth', device: str = 'cpu',
                 thresh: float = 0.4, pad_ratio: float = 0.05,
                 backend: str = 'torch', onnx_path: str = 'ecpose_m_o3652coco.onnx',
                 intra_threads: int = 4):
        self.thresh = thresh
        self.pad_ratio = pad_ratio
        self.backend = backend
        self.tf = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if backend == 'onnx':
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = intra_threads
            self.session = ort.InferenceSession(onnx_path, sess_options=so,
                                                providers=['CPUExecutionProvider'])
            self.model = None
            return
        if backend == 'openvino':
            import openvino as ov
            core = ov.Core()
            self.ov_model = core.compile_model(onnx_path.replace('.onnx', '.xml'), 'CPU')
            self.ov_out = [self.ov_model.output(i) for i in range(3)]
            self.model = None
            return
        self.device = torch.device(device)
        cfg = YAMLConfig(str(CONFIG), resume=weights)
        cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
        ckpt = torch.load(weights, map_location='cpu')
        state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
        cfg.model.load_state_dict(state)

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)

        self.model = _M().to(self.device).eval()
        self.size = tuple(cfg.yaml_cfg['eval_spatial_size'])  # (h, w) = (640, 640)

    def _normalize(self, scores, labels, kps, w: int, h: int,
                   timestamp: float, frame_id: int) -> list[Detection]:
        out: list[Detection] = []
        for s, lb, kp in zip(scores, labels, kps):
            if float(s) < self.thresh or int(lb) != PERSON_LABEL:
                continue
            kp = np.asarray(kp, dtype=np.float32).reshape(17, 2)
            x1, y1 = float(kp[:, 0].min()), float(kp[:, 1].min())
            x2, y2 = float(kp[:, 0].max()), float(kp[:, 1].max())
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            pad = self.pad_ratio
            x1 = max(0.0, x1 - bw * pad); y1 = max(0.0, y1 - bh * pad)
            x2 = min(float(w - 1), x2 + bw * pad); y2 = min(float(h - 1), y2 + bh * pad)
            kpts3 = np.concatenate([kp, np.ones((17, 1), dtype=np.float32)], axis=1)
            out.append(Detection(bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                                 score=float(s), keypoints=kpts3,
                                 frame_id=frame_id, timestamp=timestamp))
        # highest-score first (helps greedy association + TAI NMS)
        out.sort(key=lambda d: d.score, reverse=True)
        return out

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, timestamp: float, frame_id: int) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.tf(img).unsqueeze(0)
        if self.backend == 'onnx':
            arr = tensor.numpy().astype(np.float32)
            size = np.array([[img.size[0], img.size[1]]], dtype=np.int64)
            scores, labels, kps = self.session.run(
                None, {'images': arr, 'orig_target_sizes': size})
            return self._normalize(scores[0], labels[0], kps[0], w, h,
                                   timestamp, frame_id)
        if self.backend == 'openvino':
            arr = tensor.numpy().astype(np.float32)
            size = np.array([[img.size[0], img.size[1]]], dtype=np.int64)
            r = self.ov_model({'images': arr, 'orig_target_sizes': size})
            return self._normalize(r[self.ov_out[0]][0], r[self.ov_out[1]][0],
                                   r[self.ov_out[2]][0], w, h, timestamp, frame_id)
        tensor = tensor.to(self.device)
        sizes = torch.tensor([[img.size[0], img.size[1]]], device=self.device)
        scores, labels, kps = self.model(tensor, sizes)
        return self._normalize(scores[0].detach().cpu().numpy(),
                               labels[0].detach().cpu().numpy(),
                               kps[0].detach().cpu().numpy(),
                               w, h, timestamp, frame_id)
