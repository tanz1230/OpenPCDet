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
    parser.add_argument('--save_to_file', type=str, required=True, help='Path to save GT CSV file')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to save GT CSV file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem

    #print(cfg)
    np.random.seed(1024)

    return args, cfg


def main():
    # ─── 1. Load your dataset config ────────────────────────────────────────────
    args, cfg = parse_config()
    args.log_dir = Path(args.log_dir)
    log_file = args.log_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # ─── 2. Build the test “dataset” (we only use .infos, no model) ─────────────
    # batch_size and workers don't matter since we won't iterate the loader
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=1,
        training=False,
        logger=logger
    )

    # ─── 3. Walk through infos and pull out GT annotations ─────────────────────
    records = []
    for info in test_set.infos:
        # identifiers
        frame_id     = os.path.basename(info['lidar_path']).split('.')[0]
        sample_token = info.get('token', None)
        timestamp    = info.get('timestamp', None)  # seconds

        # GT arrays
        gt_boxes = info['gt_boxes']            # (N,9)
        gt_vels  = info.get('gt_boxes_velocity')  # (N,3), optional
        gt_names = info['gt_names']            # (N,)
        car_pose = info.get('car_from_global', None)
        if car_pose is not None:
            ego_x, ego_y, ego_z = car_pose[:3, 3]
        else:
            ego_x = ego_y = ego_z = None

        for i, name in enumerate(gt_names):
            # unpack: x,y,z, dx,dy,dz, yaw, vx, vy
            x, y, z, dx, dy, dz, yaw, vx, vy = gt_boxes[i]
            rec = {
                'frame_id':    frame_id,
                'sample_token': sample_token,
                'timestamp':    timestamp,
                'ego_x':        float(ego_x) if ego_x is not None else None,
                'ego_y':        float(ego_y) if ego_y is not None else None,
                'ego_z':        float(ego_z) if ego_z is not None else None,
                'x':            float(x),
                'y':            float(y),
                'z':            float(z),
                'l':            float(dx),
                'w':            float(dy),
                'h':            float(dz),
                'heading':      float(yaw),
                'label':        name,
                'vx':           float(vx),
                'vy':           float(vy),
            }
            # if you also want vz:
            if gt_vels is not None:
                rec['vz'] = float(gt_vels[i][2])
            records.append(rec)

    # ─── 4. Save to CSV or pickle ───────────────────────────────────────────────
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.save_to_file, index=False) #'/home/idiot/Research/LIDAR_Error_Modelling/nuscenes_gt_annotations.csv'
    print(f"Saved {len(df)} GT boxes to {args.save_to_file}")

if __name__ == '__main__':
    main()
