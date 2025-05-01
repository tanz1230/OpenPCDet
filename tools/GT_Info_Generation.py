#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 01:21:03 2025

@author: idiot
"""
import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--save_to_file', action='store_true', required=True, default='/home/idiot/Research/LIDAR_Error_Modelling/nuscenes_gt_annotations.csv', help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    # ─── 1. Load your dataset config ────────────────────────────────────────────
    args, cfg = parse_config()

    # ─── 2. Build the test “dataset” (we only use .infos, no model) ─────────────
    # batch_size and workers don't matter since we won't iterate the loader
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=1,
        training=False
    )

    # ─── 3. Walk through infos and pull out GT annotations ─────────────────────
    records = []
    for info in test_set.infos:
        # these keys come straight from the .pkl infos that OpenPCDet generated:
        frame_id     = os.path.basename(info['lidar_path']).split('.')[0]
        sample_token = info.get('token', None)
        timestamp    = info.get('timestamp', None)        # in microseconds

        # ‘annos’ holds your ground‐truth boxes and class names
        annos     = info['annos']
        gt_boxes  = annos['gt_boxes']   # (N,7) → [x, y, z, dx, dy, dz, yaw]
        gt_names  = annos['name']       # list of length N

        for i, name in enumerate(gt_names):
            x, y, z, dx, dy, dz, yaw = gt_boxes[i]
            rec = {
                'frame_id':    frame_id,
                'sample_token': sample_token,
                'timestamp':    timestamp,
                'x':      float(x),
                'y':      float(y),
                'z':      float(z),
                'l':      float(dx),   # length (along X in NuScenes)
                'w':      float(dy),   # width  (along Y)
                'h':      float(dz),   # height
                'heading': float(yaw),
                'label':   name
            }
            records.append(rec)

    # ─── 4. Save to CSV or pickle ───────────────────────────────────────────────
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.save_to_file, index=False) #'/home/idiot/Research/LIDAR_Error_Modelling/nuscenes_gt_annotations.csv'
    print(f"Saved {len(df)} GT boxes to {args.save_to_file}")

if __name__ == '__main__':
    main()
