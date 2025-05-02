#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 01:21:03 2025

@author: idiot
"""

import numpy as np

def match_detections_to_gt(
    gt_boxes,          # (N_gt, 7+) arrays: [x, y, z, l, w, h, yaw, …]
    pred_boxes,        # (N_pred, 7+) same format
    gt_labels,         # list/array of length N_gt
    pred_labels,       # list/array of length N_pred
    dist_threshold=2.0 # matching threshold in meters
):
    """
    Returns:
      gt_matched  : (N_gt,) boolean mask — True if GT was matched by some pred
      pred_matched: (N_pred,) boolean mask — True if pred was matched to some GT

    Matching is greedy: we compute all (GT, pred) pairs of the same class
    whose 2D center‐distance ≤ dist_threshold, sort by increasing distance,
    and then assign matches one‐to‐one.
    """
    N_gt   = len(gt_boxes)
    N_pred = len(pred_boxes)

    gt_matched   = np.zeros(N_gt,   dtype=bool)
    pred_matched = np.zeros(N_pred, dtype=bool)

    # 1) collect all eligible pairs
    pairs = []
    for i in range(N_gt):
        for j in range(N_pred):
            if gt_labels[i] != pred_labels[j]:
                continue
            # 2D center distance
            dx = gt_boxes[i][0] - pred_boxes[j][0]
            dy = gt_boxes[i][1] - pred_boxes[j][1]
            dist = np.hypot(dx, dy)
            if dist <= dist_threshold:
                pairs.append((dist, i, j))

    # 2) sort by distance (smallest first)
    pairs.sort(key=lambda x: x[0])

    # 3) greedy one-to-one assignment
    for dist, i, j in pairs:
        if not gt_matched[i] and not pred_matched[j]:
            gt_matched[i]   = True
            pred_matched[j] = True

    return gt_matched, pred_matched