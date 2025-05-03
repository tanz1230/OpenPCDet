#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 01:21:03 2025

@author: idiot
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 02:XX 2025

@author: idiot

Generate CenterPoint test‐time predictions into a CSV file with
frame_id, sample_token, timestamp, lidar_path, ego pose, and box attributes.
"""
import _init_path
import argparse
import datetime
import os
from pathlib import Path

import torch
import pandas as pd
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file',    type=str, required=True,
                        help='specify the config file for testing')
    parser.add_argument('--ckpt',        type=str, required=True,
                        help='path to model checkpoint (.pth)')
    parser.add_argument('--batch_size',  type=int, default=1,
                        help='batch size for inference')
    parser.add_argument('--workers',     type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--save_to_file',type=str, required=True,
                        help='Path to save prediction CSV')
    parser.add_argument('--log_dir',     type=str, required=True,
                        help='Directory to save log file')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    return args, cfg

def main():
    # 1. Parse config & set up logger
    args, cfg = parse_config()
    args.log_dir = Path(args.log_dir)
    log_file = args.log_dir / f'log_pred_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 2. Build test dataset & loader
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        training=False,
        logger=logger
    )

    # 3. Build network & load checkpoint
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()

    # 4. Inference loop & collect records
    records = []
    sample_i = 0
    for batch in test_loader:
        print('progress:',sample_i)
        load_data_to_gpu(batch)
        with torch.no_grad():
            pred_dicts, _ = model(batch)

        # metadata from batch and info
        frame_ids    = batch.get('frame_id', [None])
        info         = test_set.infos[sample_i]
        sample_token = info.get('token', None)
        timestamp    = info.get('timestamp', None)
        lidar_path   = info.get('lidar_path', None)

        for b_ix, preds in enumerate(pred_dicts):
            boxes  = preds['pred_boxes'].cpu().numpy()   # (N,7)
            scores = preds['pred_scores'].cpu().numpy()  # (N,)
            labels = preds['pred_labels'].cpu().numpy()  # (N,)
            vels   = preds.get('pred_vel', None)
            if vels is not None:
                vels = vels.cpu().numpy()                # (N,2)

            for idx in range(boxes.shape[0]):
                if boxes.shape[1] == 7:
                    x, y, z, l, w, h, yaw = boxes[idx]
                    vx = vy = None
                else:
                    x, y, z, l, w, h, yaw, vx, vy = boxes[idx]

                rec = {
                    'frame_id':       frame_ids[b_ix],
                    'sample_token':   sample_token,
                    'timestamp':      timestamp,
                    'lidar_path':     lidar_path,
                    'x':              float(x),
                    'y':              float(y),
                    'z':              float(z),
                    'l':              float(l),
                    'w':              float(w),
                    'h':              float(h),
                    'vx':             float(vx),
                    'vy':             float(vy),
                    'heading':        float(yaw),
                    'score':          float(scores[idx]),
                    'label':          cfg.CLASS_NAMES[labels[idx]-1]
                }
                if vels is not None:
                    rec['vx'] = float(vels[idx,0])
                    rec['vy'] = float(vels[idx,1])
                records.append(rec)

        sample_i += 1

    # 5. Save to CSV
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.save_to_file, index=False)
    logger.info(f"Saved {len(df)} predictions to {args.save_to_file}")

if __name__ == '__main__':
    main()


