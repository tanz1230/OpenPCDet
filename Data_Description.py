#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:22:21 2025

@author: idiot
"""
import numpy as np

# Load one of your point cloud files
points = np.load("/media/idiot/OS/Project/Cooperative Perception/Field Test/11042024/points/000010.npy")

points = np.fromfile("/home/idiot/Downloads/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin", dtype=np.float32)
points = points.reshape(-1, 5)  # Adjust based on the actual structure
min(points[:,3])
max(points[:,3])

# Compute the min and max for each coordinate
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()

print(f"X range: {x_min:.2f} to {x_max:.2f}")
print(f"Y range: {y_min:.2f} to {y_max:.2f}")
print(f"Z range: {z_min:.2f} to {z_max:.2f}")
