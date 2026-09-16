"""Export OSNet-x0.25 MSMT17 (torchreid) to ONNX for selective ReID.

venv usage:
    .\\.venv\\Scripts\\python.exe tools\\export_osnet_onnx.py --weights osnet_x0_25_msmt17.pth

Output osnet_x0_25_msmt17.onnx (1x3x256x128, embedding) is consumed by src/dexi/reid.py.
MSMT17 combineall weights: torchreid model zoo (Market1501 R1 59.9 / Duke R1 61.5).
"""
from __future__ import annotations

import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', default='osnet_x0_25_msmt17.onnx')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    from torchreid.reid.models import osnet
    model = osnet.osnet_x0_25(num_classes=4101, loss='softmax', pretrained=False)
    state = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.rand(1, 3, 256, 128)
    with torch.no_grad():
        feat = model(dummy)
    print('embedding dim:', tuple(feat.shape))

    torch.onnx.export(model, dummy, args.out, input_names=['images'],
                      output_names=['embedding'],
                      dynamic_axes={'images': {0: 'N'}},
                      opset_version=args.opset, do_constant_folding=True,
                      dynamo=False)
    print(f'Exported {args.out}')
    import onnx
    onnx.checker.check_model(onnx.load(args.out))
    print('ONNX check passed')


if __name__ == '__main__':
    main()
