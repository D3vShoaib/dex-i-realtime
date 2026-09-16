"""Benchmark OpenVINO CPU vs ONNX Runtime CPU: latency + parity (ECPose + OSNet)."""
import time
import numpy as np
import cv2
import onnxruntime as ort
import openvino as ov

cap = cv2.VideoCapture('test.mp4')
ret, fr = cap.read()
img = cv2.resize(fr, (640, 640)).astype(np.float32)[:, :, ::-1] / 255.0
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
arr = ((img - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
size = np.array([[640, 640]], dtype=np.int64)

print('== ECPose-M (640x640) ==')
ort_s = ort.InferenceSession('ecpose_m_o3652coco.onnx', providers=['CPUExecutionProvider'])
core = ov.Core()
ov_m = core.compile_model('ecpose_m_o3652coco.xml', 'CPU')
ort_s.run(None, {'images': arr, 'orig_target_sizes': size})  # warmup
ov_m({'images': arr, 'orig_target_sizes': size})
N = 5
t0 = time.time()
for _ in range(N):
    os_, ol, ok = ort_s.run(None, {'images': arr, 'orig_target_sizes': size})
t_ort = (time.time() - t0) / N
t0 = time.time()
for _ in range(N):
    r = ov_m({'images': arr, 'orig_target_sizes': size})
t_ov = (time.time() - t0) / N
vs, vl, vk = r[ov_m.output(0)], r[ov_m.output(1)], r[ov_m.output(2)]
print(f'ORT: {t_ort:.2f}s/frame  OV: {t_ov:.2f}s/frame  speedup: {t_ort / t_ov:.2f}x')
print('max |scores| diff:', float(np.abs(os_[0] - vs[0]).max()))
print('max |kpts| diff (px):', float(np.abs(ok[0] - vk[0]).max()))
print('n>0.4 ORT:', int((os_[0] > 0.4).sum()), 'OV:', int((vs[0] > 0.4).sum()))

print('== OSNet (256x128 crop) ==')
ort_r = ort.InferenceSession('osnet_x0_25_msmt17.onnx', providers=['CPUExecutionProvider'])
ov_r = core.compile_model('osnet_x0_25_msmt17.xml', 'CPU')
crop = cv2.resize(fr[300:550, 200:300], (128, 256)).astype(np.float32)[:, :, ::-1] / 255.0
carr = (((crop - mean) / std).transpose(2, 0, 1)[None]).astype(np.float32)
ort_r.run(None, {'images': carr})
ov_r({'images': carr})
M = 20
t0 = time.time()
for _ in range(M):
    fo = ort_r.run(None, {'images': carr})[0]
t_ort = (time.time() - t0) / M * 1000
t0 = time.time()
for _ in range(M):
    fr_ = ov_r({'images': carr})[ov_r.output(0)]
t_ov = (time.time() - t0) / M * 1000
fo = fo / np.linalg.norm(fo)
fr_ = fr_ / np.linalg.norm(fr_)
print(f'ORT: {t_ort:.1f}ms/crop  OV: {t_ov:.1f}ms/crop  speedup: {t_ort / t_ov:.2f}x')
print('cos(ORT, OV) embedding:', round(float((fo.flatten() @ fr_.flatten())), 5))
