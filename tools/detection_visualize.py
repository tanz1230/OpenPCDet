#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 02:53:11 2025

@author: idiot
"""
import open3d as o3d
import numpy as np
import os

def visualize_lidar_with_kitti_labels(lidar_npy_path, label_path):
    """
    Args:
        lidar_npy_path (str): Path to the LiDAR .npy file (N,3) or (N,4)
        label_path (str): Path to the KITTI label .txt file
    """
    # Load point cloud
    pts = np.load(lidar_npy_path)  # directly load npy
    pts_xyz = pts[:, :3]           # Only take x, y, z

    # Load labels
    boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split()
            x, y, z, l, w, h, yaw = map(float, elements[:7])
            boxes.append((x, y, z, l, w, h, yaw))

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz)

    # Create 3D boxes
    bbox_list = []
    for box in boxes:
        x, y, z, l, w, h, yaw = box
        bbox = create_open3d_bbox(x, y, z, l, w, h, yaw)
        bbox_list.append(bbox)

    # Visualize
    o3d.visualization.draw_geometries([pcd, *bbox_list])

def create_open3d_bbox(x, y, z, l, w, h, yaw):
    """Create an Open3D 3D bounding box."""
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = [x, y, z]
    bbox.extent = [l, w, h]
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    bbox.R = R
    bbox.color = (1, 0, 0)  # Red
    return bbox

# Example usage:
visualize_lidar_with_kitti_labels('/home/idiot/OpenPCDet/data/CP/points/007259.npy', '/home/idiot/OpenPCDet/data/CP/labels/007259.txt')

visualize_lidar_with_kitti_labels('/home/idiot/OpenPCDet/data/CP/points/004281.npy', '/home/idiot/OpenPCDet/data/CP/labels/004281.txt')

visualize_lidar_with_kitti_labels('/home/idiot/OpenPCDet/data/debug/points/000006.npy', '/home/idiot/OpenPCDet/data/debug/labels/000006.txt')

visualize_lidar_with_kitti_labels('/home/idiot/OpenPCDet/data/V2V4Real_tesla/points/000006.npy', '/home/idiot/OpenPCDet/data/V2V4Real_tesla/labels/000006.txt')
