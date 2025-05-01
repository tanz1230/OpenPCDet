#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 2‑D intensity video with OpenPCDet‑predicted bounding boxes.

author  –  April 2025
"""

import os, glob, cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse, torch

from pcdet.config   import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models   import build_network, load_data_to_gpu
from pcdet.utils    import common_utils


# ───────────────────────── DATASET WRAPPER ──────────────────────────────────
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, ext, logger):
        super().__init__(dataset_cfg, class_names, False, root_path, logger)
        self.ext   = ext
        self.files = (sorted(glob.glob(str(root_path / f'*{ext}')))
                      if root_path.is_dir() else [root_path])

    def __len__(self):             return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        pts = (np.fromfile(f, dtype=np.float32).reshape(-1,5)
               if self.ext == '.bin' else np.load(f))
        return self.prepare_data({'points': pts, 'frame_id': idx})


# ───────────────────────── MAIN ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file',  required=True)
    ap.add_argument('--ckpt',      required=True)
    ap.add_argument('--data_path', required=True,
                    help='file or directory with .bin / .npy frames')
    ap.add_argument('--ext',       default='.bin', choices=['.bin', '.npy'])
    ap.add_argument('--video_out', required=True)
    ap.add_argument('--fps',       type=int, default=10)
    ap.add_argument('--size',      type=int, nargs=2, default=[1200,1200],
                    metavar=('W','H'))
    ap.add_argument('--scale',     type=float, default=10.0,
                    help='metres → pixels')
    ap.add_argument('--no_boxes',  action='store_true',
                    help='do not draw predicted boxes (faster)')
    args = ap.parse_args()

    W, H  = args.size
    SCALE = args.scale

    # ── Config & logger ────────────────────────────────────────────────────
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    ds = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES,
                     Path(args.data_path), args.ext, logger)
    logger.info(f'# frames: {len(ds)}')

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), ds)
    model.load_params_from_file(args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()

    # ── Video writer ───────────────────────────────────────────────────────
    writer = cv2.VideoWriter(args.video_out,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             args.fps, (W, H))

    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_PLASMA)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(ds, desc='render')):
            batch = ds.collate_batch([sample])
            load_data_to_gpu(batch)
            preds, _ = model(batch)

            pts  = batch['points'][:,1:].cpu().numpy()        # (N,>=4)
            img  = bev_intensity_image(pts, W, H, SCALE, lut)

            if not args.no_boxes:
                boxes = preds[0]['pred_boxes'].cpu().numpy()  # (K,7)
                draw_boxes_bev(img, boxes, W, H, SCALE)

            if pts.shape[1] >= 5:
                ts = int(pts[0,4])
                cv2.putText(img, f'{ts} ms', (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            writer.write(img)

    writer.release()
    logger.info('video saved →  ' + args.video_out)


# ───────────────────────── RENDER UTILITIES ─────────────────────────────────
def bev_intensity_image(pts, W, H, scale, lut=None):
    """
    Return a BGR canvas (H×W) coloured by intensity.
    If `lut` is provided (256×1×3 BGR), use it; otherwise build a TURBO LUT
    the first time and reuse it.
    """
    u = (pts[:, 0] * scale + W/2).astype(int)
    v = (pts[:, 1] * scale + H/2).astype(int)
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    inten = pts[:, 3]
    inten_n = (inten - inten.min()) / (inten.ptp() + 1e-6)   # 0‑1
    idx = (np.sqrt(inten_n) * 255).astype(np.uint8)          # gamma 0.5

    # choose colour‑map -----------------------------------------------------
    if lut is not None:
        colours = lut[idx]                       # external palette
    else:
        if not hasattr(bev_intensity_image, '_lut'):
            turbo = cv2.applyColorMap(np.arange(256, dtype=np.uint8),
                                      cv2.COLORMAP_TURBO)
            bev_intensity_image._lut = turbo
        colours = bev_intensity_image._lut[idx]  # cached TURBO palette

    # scatter plot ----------------------------------------------------------
    img = np.zeros((H, W, 3), np.uint8)
    img[v[mask], u[mask]] = colours[mask, 0]
    img = cv2.dilate(img, np.ones((2,2), np.uint8))

    # cross‑hair ------------------------------------------------------------
    cv2.line(img, (W//2, 0), (W//2, H), (60,60,60), 1)
    cv2.line(img, (0, H//2), (W, H//2), (60,60,60), 1)
    return img



def draw_boxes_bev(img, boxes, W, H, scale):
    """Overlay green rectangles (LiDAR XY footprint) on the image."""
    for x,y,z,dx,dy,dz,yaw in boxes:
        rot = np.array([[ np.cos(yaw), -np.sin(yaw)],
                        [ np.sin(yaw),  np.cos(yaw)]])
        c   = np.array([[ dx/2,  dy/2], [ dx/2,-dy/2],
                        [-dx/2,-dy/2], [-dx/2, dy/2]])
        xy  = (c @ rot.T) + np.array([x,y])
        pix = (xy*scale + np.array([W/2,H/2])).astype(int)
        cv2.polylines(img, [pix], True, (0,255,0), 2)


# ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
