"""Extract diverse calibration frames from a video for INT8 quantization.

Uniform sampling across the whole duration (covers lighting/crowd changes).
Regenerable at any time; output dir is gitignored.

venv usage:
    .\\.venv\\Scripts\\python.exe tools\\make_calibration.py --input test.mp4 --num 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='test.mp4')
    ap.add_argument('--out-dir', default='calibration')
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--quality', type=int, default=90)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(i * total / args.num) for i in range(args.num)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(str(out / f'cal_{k:04d}.jpg'),
                    frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        saved += 1
    cap.release()
    print(f'Saved {saved}/{args.num} frames to {out}/ (total source frames: {total})')


if __name__ == '__main__':
    main()
