"""Export ECPose-M to ONNX for ONNX Runtime CPU speedup (Phase 2).

venv usage:
    .\\.venv\\Scripts\\python.exe tools\\export_ecpose_onnx.py -c third_party\\EdgeCrafter\\ecpose\\configs\\ecpose\\ecpose_m_coco.yml -r ecpose_m_o3652coco.pth --check

Mirrors upstream ecpose/tools/deployment/export_onnx.py, paths adapted to this repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'third_party' / 'EdgeCrafter' / 'ecpose'))
from engine.core import YAMLConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--opset', type=int, default=18)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--simplify', action='store_true')
    args = ap.parse_args()

    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
    ckpt = torch.load(args.resume, map_location='cpu')
    state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = Model().eval()
    img_size = cfg.yaml_cfg['eval_spatial_size']
    data = torch.rand(1, 3, *img_size)
    size = torch.tensor([[img_size[1], img_size[0]]])  # W, H like inference
    _ = model(data, size)

    out = Path(args.resume).with_suffix('.onnx')
    torch.onnx.export(model, (data, size), str(out),
                      input_names=['images', 'orig_target_sizes'],
                      output_names=['scores', 'labels', 'keypoints'],
                      dynamic_axes={'images': {0: 'N'}, 'orig_target_sizes': {0: 'N'}},
                      opset_version=args.opset, do_constant_folding=True,
                      dynamo=False)
    print(f'Exported {out}')
    if args.check:
        import onnx
        onnx.checker.check_model(onnx.load(str(out)))
        print('ONNX check passed')
    if args.simplify:
        import onnx
        import onnxsim
        simp, ok = onnxsim.simplify(str(out))
        onnx.save(simp, str(out))
        print(f'Simplified: {ok}')


if __name__ == '__main__':
    main()
