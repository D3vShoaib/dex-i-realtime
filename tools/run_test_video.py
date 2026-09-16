"""Offline demo: input MP4 -> out/annotated.mp4 + out/tracks.json.

Stack (single implementation): ECPose-M OpenVINO IR -> ultralytics TrackTrack ->
selective OSNet OpenVINO IR ReID + OKS persistent IDs.

venv usage:
    python tools\\run_test_video.py --input test_first_1min.mp4 --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from dexi.frame_source import Mp4Source
from dexi.pipeline import ECPOSE_IR, REID_IR, DexiPipeline
from dexi.vis import draw_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='test_first_1min.mp4')
    ap.add_argument('--tracker-cfg', default='configs/tracktrack_dexi.yaml')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--thresh', type=float, default=0.4)
    ap.add_argument('--out-dir', default='out')
    ap.add_argument('--no-skeleton', action='store_true')
    ap.add_argument('--ecpose-ir', default=ECPOSE_IR,
                    help='ECPose-M OpenVINO IR (.xml)')
    ap.add_argument('--no-reid', action='store_true',
                    help='disable selective OSNet ReID (tracker IDs only)')
    ap.add_argument('--reid-ir', default=REID_IR,
                    help='OSNet OpenVINO IR (.xml) for selective ReID')
    ap.add_argument('--threads', type=int, default=None,
                    help='OpenVINO CPU inference threads (default: OpenVINO auto)')
    ap.add_argument('--rebind-thr', type=float, default=0.55)
    ap.add_argument('--lost-ttl', type=int, default=300,
                    help='frames to retain a lost identity (default: 300 = 60s at 5 FPS)')
    ap.add_argument('--gallery-size', type=int, default=5,
                    help='appearance embeddings retained per identity')
    ap.add_argument('--reid-refresh', type=int, default=5,
                    help='frames between global identity assignments')
    ap.add_argument('--spatial-weight', type=float, default=0.10,
                    help='location contribution to global assignment score')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = DexiPipeline(det_thresh=args.thresh, tracker_cfg=args.tracker_cfg,
                        ecpose_ir=args.ecpose_ir, reid=not args.no_reid,
                        reid_ir=args.reid_ir, rebind_thr=args.rebind_thr,
                        lost_ttl=args.lost_ttl, gallery_size=args.gallery_size,
                        reid_refresh=args.reid_refresh, spatial_w=args.spatial_weight,
                        num_threads=args.threads)
    src = Mp4Source(args.input, limit=args.limit, stride=args.stride)

    cap = cv2.VideoCapture(args.input)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    cap.release()

    writer = cv2.VideoWriter(str(out_dir / 'annotated.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps / args.stride, (W, H))
    records: list[dict] = []
    t0 = time.time()
    n_det = n_trk = 0
    for frame, dets, tracks, ts, fid in pipe.run(src):
        n_det += len(dets)
        n_trk += len(tracks)
        vis = draw_tracks(frame, tracks, draw_skeleton=not args.no_skeleton)
        writer.write(vis)
        records.append({
            'frame_id': fid, 'timestamp': ts,
            'detections': [{'bbox': d.bbox_xyxy.tolist(), 'score': d.score,
                            'keypoints': d.keypoints.tolist()} for d in dets],
            'tracks': [{'id': t.track_id, 'tid': t.tid, 'bbox': t.bbox_xyxy.tolist(),
                        'score': t.score,
                        'keypoints': None if t.keypoints is None else t.keypoints.tolist()}
                       for t in tracks],
        })
        done = len(records)
        if done == 1 or done % 5 == 0 or done == args.limit:
            el = time.time() - t0
            print(f'[{done}/{args.limit}] fid={fid} det={len(dets)} trk={len(tracks)} '
                  f'{el / done:.1f}s/frame', flush=True)
    writer.release()
    src.close()
    with open(out_dir / 'tracks.json', 'w') as f:
        json.dump(records, f)
    el = time.time() - t0
    print(f'Done: {len(records)} frames, {n_det} dets, {n_trk} track-obs, '
          f'{el:.1f}s total ({el / max(1, len(records)):.2f}s/frame)')
    m = pipe.manager
    if m.enable:
        print(f'ReID: {m.reid_calls} encoder calls over {m.frames} frames '
              f'({m.reid_calls / max(1, m.frames):.2f}/frame), {m.rebinds} rebinds')
    print(f'Wrote {out_dir / "annotated.mp4"} and {out_dir / "tracks.json"}')


if __name__ == '__main__':
    main()
