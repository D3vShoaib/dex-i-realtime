"""Benchmark ECPose-M + OSNet on Vega 8 via DirectML vs CPU (isolated .venv-dml)."""
import time
import numpy as np
import cv2
import onnxruntime as ort

print('providers:', ort.get_available_providers())
DML = 'DmlExecutionProvider' in ort.get_available_providers()

cap = cv2.VideoCapture('test.mp4')
ret, fr = cap.read()
img = cv2.resize(fr, (640, 640)).astype(np.float32)[:, :, ::-1] / 255.0
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
arr = ((img - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
size = np.array([[640, 640]], dtype=np.int64)

print('== ECPose-M ==')
results = {}
for name, prov in [('CPU', ['CPUExecutionProvider'])] + ([('DML-Vega8', ['DmlExecutionProvider'])] if DML else []):
    try:
        s = ort.InferenceSession('ecpose_m_o3652coco.onnx', providers=prov)
        print(name, 'bound to:', s.get_providers())
        s.run(None, {'images': arr, 'orig_target_sizes': size})
        t0 = time.time()
        N = 3
        for _ in range(N):
            o = s.run(None, {'images': arr, 'orig_target_sizes': size})
        dt = (time.time() - t0) / N
        results[name] = (dt, o)
        print(f'{name}: {dt:.2f}s/frame, n>0.4={int((o[0][0] > 0.4).sum())}')
    except Exception as e:
        print(f'{name} FAILED: {str(e)[:300]}')

print('== OSNet ==')
crop = cv2.resize(fr[300:550, 200:300], (128, 256)).astype(np.float32)[:, :, ::-1] / 255.0
carr = (((crop - mean) / std).transpose(2, 0, 1)[None]).astype(np.float32)
for name, prov in [('CPU', ['CPUExecutionProvider'])] + ([('DML-Vega8', ['DmlExecutionProvider'])] if DML else []):
    try:
        s = ort.InferenceSession('osnet_x0_25_msmt17.onnx', providers=prov)
        s.run(None, {'images': carr})
        M = 10
        t0 = time.time()
        for _ in range(M):
            s.run(None, {'images': carr})
        print(f'{name}: {(time.time() - t0) / M * 1000:.1f}ms/crop')
    except Exception as e:
        print(f'{name} FAILED: {str(e)[:300]}')
