"""ECPose-M O365 adapter: Frame -> normalized Detection objects.

Single implementation: the FP32 OpenVINO IR (models/ecpose_m_o3652coco.xml/.bin)
is compiled once on CPU and every frame goes through it. Upstream weights come
from EdgeCrafter ecpose (weights -> models/ecpose_m_o3652coco.onnx -> IR via
`ov.convert_model`).

Deploy-mode outputs are (scores[N,Q], labels[N,Q], keypoints[N,Q,17,2]) with NO
explicit bbox, so bbox_xyxy is derived from the keypoint envelope + padding.
"""
from __future__ import annotations

import cv2
import numpy as np
import openvino as ov

from .types import Detection

PERSON_LABEL = 1
IR_PATH = 'models/ecpose_m_o3652coco.xml'
INPUT_SIZE = 640  # square eval_spatial_size, from ecpose_m_coco.yml
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class ECPoseDetector:
    def __init__(self, ir_path: str = IR_PATH, thresh: float = 0.4,
                 pad_ratio: float = 0.05, device: str = 'CPU',
                 num_threads: int | None = None):
        self.thresh = thresh
        self.pad_ratio = pad_ratio
        self.ir_path = ir_path
        props = {} if num_threads is None else {ov.properties.inference_num_threads: num_threads}
        self.model = ov.Core().compile_model(ir_path, device, props)
        self.out = [self.model.output(i) for i in range(3)]  # scores, labels, keypoints

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 HxWx3 -> NCHW float32 (1,3,640,640): RGB + ImageNet norm."""
        img = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

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

    def infer(self, frame_bgr: np.ndarray, timestamp: float, frame_id: int) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        arr = self._preprocess(frame_bgr)
        size = np.array([[w, h]], dtype=np.int64)  # orig_target_sizes = (W, H)
        r = self.model({'images': arr, 'orig_target_sizes': size})
        return self._normalize(r[self.out[0]][0], r[self.out[1]][0],
                               r[self.out[2]][0], w, h, timestamp, frame_id)
