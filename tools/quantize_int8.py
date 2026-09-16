"""INT8-quantize the ECPose-M IR with NNCF using calibration/ frames.

venv usage:
    .\\.venv\\Scripts\\python.exe tools\\quantize_int8.py --model ecpose_m_o3652coco.xml --subset 100

Output: ecpose_m_o3652coco_int8.xml/.bin + latency/accuracy report vs FP32.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import nncf


def preprocess(path: str):
    fr = cv2.imread(path)
    img = cv2.resize(fr, (640, 640)).astype(np.float32)[:, :, ::-1] / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    arr = ((img - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    return {'images': arr, 'orig_target_sizes': np.array([[640, 640]], dtype=np.int64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ecpose_m_o3652coco.xml')
    ap.add_argument('--cal-dir', default='calibration')
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    paths = sorted(str(p) for p in Path(args.cal_dir).glob('*.jpg'))[:args.subset]
    print(f'Calibrating on {len(paths)} frames')
    core = ov.Core()
    model = core.read_model(args.model)
    ds = nncf.Dataset(paths, preprocess)
    qmodel = nncf.quantize(model, ds, preset=nncf.QuantizationPreset.MIXED)
    out = args.out or str(Path(args.model).with_name(Path(args.model).stem + '_int8.xml'))
    ov.save_model(qmodel, out)
    print(f'Saved {out}')

    # latency + accuracy parity on live video frames (not calibration stills)
    import os
    tmp = [f'calibration/check_{i}.jpg' for i in range(5)]
    cap = cv2.VideoCapture('test.mp4')
    for t in tmp:
        ret, fr = cap.read()
        cv2.imwrite(t, fr)
    cap.release()
    try:
        fp32 = core.compile_model(args.model, 'CPU')
        int8 = core.compile_model(out, 'CPU')
        fp32({'images': preprocess(tmp[0])['images'],
              'orig_target_sizes': preprocess(tmp[0])['orig_target_sizes']})
        int8({'images': preprocess(tmp[0])['images'],
              'orig_target_sizes': preprocess(tmp[0])['orig_target_sizes']})
        for name, m in (('FP32', fp32), ('INT8', int8)):
            t0 = time.time()
            for t in tmp:
                d = preprocess(t)
                m(d)
            print(f'{name}: {(time.time() - t0) / len(tmp):.2f}s/frame')
        outs = []
        for m in (fp32, int8):
            per = []
            for t in tmp:
                d = preprocess(t)
                r = m(d)
                per.append((r[m.output(0)][0], r[m.output(2)][0]))
            outs.append(per)
        for i, ((s0, k0), (s1, k1)) in enumerate(zip(*outs)):
            print(f'frame {i}: n>0.4 fp32={int((s0 > 0.4).sum())} int8={int((s1 > 0.4).sum())} '
                  f'max|ds|={np.abs(s0 - s1).max():.4f} max|dk|={np.abs(k0 - k1).max():.2f}px')
    finally:
        for t in tmp:
            os.remove(t)


if __name__ == '__main__':
    main()
