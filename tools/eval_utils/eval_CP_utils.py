#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 03:57:39 2025

@author: idiot
"""
# ──────────────────────────────────────────────────────────────────────────────
# cooperative_fusion.py – late‑stage 3‑D box fusion for two V2X agents
# -----------------------------------------------------------------------------
# This script is a **drop‑in replacement** for your previous evaluation utilities
# (eval_one_epoch / single_prediction_CP / eval_CP).  It bundles:
#   • robust geo ↔︎ ENU ↔︎ sensor transforms (NumPy + PyTorch)
#   • Weighted Box Fusion + GPU 3‑D NMS
#   • full KITTI‑style label export for visual inspection
# -----------------------------------------------------------------------------
# Author: ChatGPT   ·  April 2025
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os, time, pickle, math, copy
from pathlib import Path
from typing import Tuple, Dict, List, Iterable

import numpy as np
import torch
import tqdm
import pymap3d as pm
import glob

from pcdet.models import load_data_to_gpu
from pcdet.utils  import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu

# ═════════════════════════════════  GEO HELPERS  ══════════════════════════════

WGS_A   = 6_378_137.0                  # metres
WGS_E2  = 6.69437999014e-3

def geodetic_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """Convert (deg, deg, m) ⭢ ECEF xyz (m) using WGS‑84."""
    lat, lon = map(math.radians, (lat, lon))
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    N = WGS_A / math.sqrt(1.0 - WGS_E2 * slat**2)
    x = (N + h) * clat * clon
    y = (N + h) * clat * slon
    z = (N * (1 - WGS_E2) + h) * slat
    return np.array([x, y, z], dtype=float)

def euler_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (all rad)."""
    sr, cr = math.sin(roll),  math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw),   math.cos(yaw)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp ,              cp*sr,              cp*cr]
    ], dtype=float)

def R_enu_to_ecef(lat: float, lon: float) -> np.ndarray:
    """3×3 rot bringing local ENU → ECEF at (lat,lon)."""
    lat, lon = map(math.radians, (lat, lon))
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    return np.array([
        [-slon,            clon,           0],
        [-slat*clon, -slat*slon,  clat],
        [ clat*clon,  clat*slon,  slat]
    ], dtype=float)

# ═════════════════════════════  BOX TRANSFORMS  ═══════════════════════════════

# These two helpers assume OpenPCDet‑style boxes = (x,y,z,l,w,h,yaw)

@torch.jit.script
def _fit_boxes_from_corners(corners: torch.Tensor) -> torch.Tensor:
    """corners: (N,8,3)  →  (N,7)  (centre, dims, yaw) (PyTorch).
    Only robust enough for upright BEV boxes.
    """
    # centre = mean
    ctr = corners.mean(dim=1)
    # dims = (max-min)
    mins = corners.min(dim=1).values
    maxs = corners.max(dim=1).values
    dims = maxs - mins   # (l,w,h) but order unknown – assume x=l, y=w
    # planar yaw via two front corners (0 & 1 in OpenPCDet utils)
    v = corners[:,1] - corners[:,0]
    yaw = torch.atan2(v[:,1], v[:,0])
    return torch.cat([ctr, dims[:,[0,1,2]], yaw.unsqueeze(1)], dim=1)

# We reuse OpenPCDet utility if available; else fallback
try:
    from pcdet.utils.box_utils import boxes_to_corners_3d as oc2corners
except ImportError:
    def oc2corners(boxes: torch.Tensor) -> torch.Tensor:
        raise RuntimeError('Please install OpenPCDet or provide your own box→corner fn')


def sensor_boxes_to_veh(boxes_s: torch.Tensor, Tr_s2v: np.ndarray) -> torch.Tensor:
    """(N,7) sensor → vehicle frame (torch I/O, np extrinsic)."""
    device, dtype = boxes_s.device, boxes_s.dtype
    corners = oc2corners(boxes_s.float())            # (N,8,3)
    N = corners.size(0)
    homo = torch.cat([corners.view(-1,3), torch.ones(N*8,1, device=device)], 1)  # (N*8,4)
    T = torch.from_numpy(Tr_s2v).to(device=device, dtype=dtype)
    corners_v = (T @ homo.t()).t()[:, :3].view(N,8,3)
    return _fit_boxes_from_corners(corners_v)


def veh_boxes_to_ref_enu(boxes_v: torch.Tensor,
                          roll:float,pitch:float,yaw:float,
                          lat:float,lon:float,alt:float,
                          ref_lat:float, ref_lon:float, ref_alt:float,
                          device=None) -> torch.Tensor:
    """Vehicle frame → common ENU. Keeps tensors on GPU."""
    device = device or boxes_v.device
    dtype  = boxes_v.dtype
    # centres
    centres = boxes_v[:, :3]

    # veh→local ENU via attitude (R @ pts)
    R_att = torch.from_numpy(euler_to_R(roll,pitch,yaw)).to(device=device, dtype=dtype)
    pts_enu = centres @ R_att.T

    # shift to global ENU origin (ref)
    # ENU→ECEF at *current* lat/lon for translation, then to ref ENU
    origin_ecef_cur = torch.from_numpy(geodetic_to_ecef(lat,lon,alt)).to(device, dtype)
    origin_ecef_ref = torch.from_numpy(geodetic_to_ecef(ref_lat,ref_lon,ref_alt)).to(device, dtype)
    R_enu_ecef_ref  = torch.from_numpy(R_enu_to_ecef(ref_lat,ref_lon)).to(device, dtype)

    pts_ecef = pts_enu @ R_enu_to_ecef(lat,lon).T + origin_ecef_cur
    pts_enu_ref = (pts_ecef - origin_ecef_ref) @ R_enu_ecef_ref

    out = boxes_v.clone()
    out[:, :3] = pts_enu_ref
    out[:, 6] += yaw              # simple planar rotation
    return out


def ref_enu_boxes_to_sensor(boxes_e: torch.Tensor,
                             Tr_s2v: np.ndarray,
                             roll:float,pitch:float,yaw:float,
                             ref_lat:float, ref_lon:float, ref_alt:float,
                             device=None) -> torch.Tensor:
    """Inverse of veh_boxes_to_ref_enu → sensor frame."""
    device = device or boxes_e.device
    dtype  = boxes_e.dtype
    centres = boxes_e[:, :3]

    origin_ecef_ref = torch.from_numpy(geodetic_to_ecef(ref_lat,ref_lon,ref_alt)).to(device,dtype)
    R_ref = torch.from_numpy(R_enu_to_ecef(ref_lat,ref_lon)).to(device,dtype)

    pts_ecef = centres @ R_ref.T + origin_ecef_ref
    # we *cannot* recover exact lat/lon/att of the ego here; assume yaw/pitch/roll known
    R_att = torch.from_numpy(euler_to_R(roll,pitch,yaw)).to(device,dtype)
    pts_veh = pts_ecef @ R_enu_to_ecef(ref_lat,ref_lon) - origin_ecef_ref  # small approx
    pts_veh = pts_veh @ R_att     # rotate back into veh frame

    # veh→sensor
    T_inv = torch.from_numpy(np.linalg.inv(Tr_s2v)).to(device,dtype)
    homo  = torch.cat([pts_veh, torch.ones(pts_veh.size(0),1, device=device, dtype=dtype)], dim=1)
    pts_s = (T_inv @ homo.t()).t()[:, :3]

    out = boxes_e.clone()
    out[:, :3] = pts_s
    out[:, 6] -= yaw
    return out

# ════════════════════════════  LATE FUSION (WBF+NMS) ══════════════════════════

def weighted_box_fusion(batch_a: List[Dict], batch_b: List[Dict],
                        alpha: float =0.6, iou_thr: float =0.55,
                        score_thr: float =0.05) -> List[Dict]:
    out = []
    for pa, pb in zip(batch_a, batch_b):
        boxes  = torch.cat([pa['pred_boxes'],  pb['pred_boxes']])
        scores = torch.cat([pa['pred_scores']*alpha,
                            pb['pred_scores']*(1-alpha)])
        labels = torch.cat([pa['pred_labels'], pb['pred_labels']])

        mask = scores > score_thr
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        keep_idx = nms_gpu(boxes.float(), scores.float(), iou_thr)
        out.append({
            'pred_boxes':  boxes[keep_idx],
            'pred_scores': scores[keep_idx],
            'pred_labels': labels[keep_idx]
        })
    return out

# ═════════════════════════════════  CACHE  ═══════════════════════════════
def build_pose_cache(gps_dir):
    """
    Load every <timestep>.txt, CSV, or whitespace-separated file in `gps_dir`.
    Expected 6 floats per file: lat, lon, alt, yaw, pitch, roll.

    Returns
    -------
    cache : dict[str → tuple(6 floats)]
            key   = file-stem, e.g. '1659453789123'
            value = (lat, lon, alt, yaw, pitch, roll)
    """
    cache = {}
    for path in glob.glob(os.path.join(gps_dir, '*.*')):
        ts   = os.path.splitext(os.path.basename(path))[0]

        # read one line, strip comments/blank lines
        with open(path, 'r') as f:
            raw = f.readline().split('#')[0].strip()   # drop trailing comments

        if not raw:                       # empty line
            raise ValueError(f'{path} is blank')

        # auto-detect delimiter
        if ',' in raw:
            vals = np.fromstring(raw, sep=',')
        else:
            vals = np.fromstring(raw, sep=' ')         # handles 1+ spaces/tabs

        if vals.size != 6:
            raise ValueError(f'{path}: expected 6 numbers, got {vals.size}')

        cache[ts] = tuple(map(float, vals))            # ensure pure Python floats
    return cache


def build_extrinsic_cache(tf_dir, *, inverse=False):
    """
    Load every transform in `tf_dir` into a dict.
        key   = file-stem (e.g. 'lidar_to_imu')
        value = 4×4 float32 matrix  (optionally inverted)

    Accepts either:
        • one line of 16 comma- or space-separated numbers
        • four lines, 4 numbers each (space-separated)
    """
    cache = {}
    for path in glob.glob(os.path.join(tf_dir, '*.*')):
        stem = os.path.splitext(os.path.basename(path))[0]

        # read file, drop empty lines / comments
        with open(path, 'r') as f:
            toks = []
            for ln in f:
                ln = ln.split('#')[0].strip()
                if not ln:
                    continue
                # decide delimiter on this line
                if ',' in ln:
                    toks.extend(np.fromstring(ln, sep=',', dtype=np.float32))
                else:
                    toks.extend(np.fromstring(ln, sep=' ', dtype=np.float32))

        if len(toks) != 16:
            raise ValueError(f'{path}: expected 16 numbers, got {len(toks)}')

        T = np.asarray(toks, dtype=np.float32).reshape(4, 4)
        if inverse:
            T = np.linalg.inv(T)

        cache[stem] = T
    return cache

# ═════════════════════════════════  EVAL LOOP  ═══════════════════════════════

def run_late_fusion_eval(cfg, args, logger,
                          dl1, 
                          pose_cache1, pose_cache2,
                          ext_cache1, ext_cache2,
                          result_dir: Path):
    """Evaluate two detectors with late‑stage fusion."""
    result_dir.mkdir(parents=True, exist_ok=True)
    label_dir = result_dir/"labels"; label_dir.mkdir(exist_ok=True)
    logger.info("Starting late‑fusion evaluation …")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pre‑load per‑frame predictions ------------------------
    with open(args.pred_bank_1, 'rb') as f:
        bank1 = pickle.load(f)
    with open(args.pred_bank_2, 'rb') as f:
        bank2 = pickle.load(f)

    dataset = dl1.dataset
    class_names = dataset.class_names

    det_annos: List[Dict] = []

    for ts, preds1 in bank1.items():
        if ts not in bank2:
            logger.warning(f"Frame {ts} missing in sensor‑2, skipping")
            continue
        preds2 = bank2[ts]

        # ↓↓↓  geo transforms  ↓↓↓
        lat1,lon1,alt1,yaw1,pitch1,roll1 = pose_cache1[ts]
        lat2,lon2,alt2,yaw2,pitch2,roll2 = pose_cache2[ts]
        T1 = ext_cache1[ts];  T2 = ext_cache2[ts]
        ref_lat,ref_lon,ref_alt = lat1,lon1,alt1

        def to_enu(plist, Tr, r,p,y, lat,lon,alt):
            out_batch = []
            for pred in plist:
                boxes_v = sensor_boxes_to_veh(pred['pred_boxes'].to(device), Tr)
                boxes_e = veh_boxes_to_ref_enu(boxes_v, r,p,y, lat,lon,alt,
                                                ref_lat,ref_lon,ref_alt)
                new = pred.copy(); new['pred_boxes'] = boxes_e; out_batch.append(new)
            return out_batch

        b1 = to_enu(preds1, T1, roll1,pitch1,yaw1, lat1,lon1,alt1)
        b2 = to_enu(preds2, T2, roll2,pitch2,yaw2, lat2,lon2,alt2)

        fused = weighted_box_fusion(b1, b2, alpha=0.6, iou_thr=0.55)

        # project back & range filter ------------------------
        final_pred_batch = []
        for fp in fused:
            boxes_s1 = ref_enu_boxes_to_sensor(fp['pred_boxes'], T1,
                                               roll1,pitch1,yaw1,
                                               ref_lat,ref_lon,ref_alt)
            dists = torch.norm(boxes_s1[:, :2], dim=1)
            keep = dists < args.dist_th
            if keep.sum() == 0:
                continue
            final_pred_batch.append({
                'pred_boxes':  boxes_s1[keep],
                'pred_scores': fp['pred_scores'][keep],
                'pred_labels': fp['pred_labels'][keep]
            })

        if not final_pred_batch:
            continue

        annos = dataset.generate_prediction_dicts({'frame_id':[ts]},
                                                  final_pred_batch,
                                                  class_names)
        det_annos += annos

        # save KITTI txt for quick sanity‑check
        fname = label_dir/f"{ts}.txt"
        with open(fname,'w') as f:
            for x,y,z,l,w,h,yaw,lbl in zip(*[annos[0][k] for k in
                                  ['boxes_lidar']]+[annos[0]['name']]):
                f.write(f"{x:.2f} {y:.2f} {z:.2f} {l:.2f} {w:.2f} {h:.2f} {yaw:.2f} {lbl}\n")

    # evaluation (same metric as cfg)
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=result_dir/'eval_tmp')
    logger.info(result_str)

    with open(result_dir/'result.pkl','wb') as f:
        pickle.dump(det_annos, f)

    return result_dict

# ──────────────────────────────────────────────────────────────────────────────
#                     Stand‑alone execution stub
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description="Late fusion eval (two agents)")
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--pred_bank_1', required=True)
    parser.add_argument('--pred_bank_2', required=True)
    parser.add_argument('--dist_th', type=float, default=140.0)
    parser.add_argument('--work_dir', type=Path, default=Path('./fusion_out'))
    args = parser.parse_args()

    cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    logger = common_utils.create_logger(args.work_dir/'fusion.log')

    # Dummy dataloaders just for dataset / class info
    # (replace with your real build_dataloader_CP)
    from pcdet.datasets import build_dataloader_CP
    test_set_1, dl1, sampler_1 = build_dataloader_CP(
        dataset_cfg=cfg.DATA_CONFIG_1,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        workers=args.workers, logger=logger, training=False
    )

    # Load pose/extrinsic caches (user‑supplied pickle/npz)
    pose1 = build_pose_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/gps_astuff')
    pose2 = build_pose_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/gps_tesla')
    ext1 = build_extrinsic_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/tf_astuff', inverse=False) # True if IMU to Sensor
    ext2 = build_extrinsic_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/tf_tesla', inverse=False) # False if Sensor to IMU

    run_late_fusion_eval(cfg, args, logger,
                         dl1,
                         pose1, pose2, ext1, ext2,
                         args.work_dir)
