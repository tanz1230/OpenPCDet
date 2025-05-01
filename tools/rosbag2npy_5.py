#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 03:38:53 2025

@author: idiot
"""
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os

# === Configuration ===
bag_file = '/media/idiot/T7 Shield/LiDAR Data/CP_lexus/10_17_2024_rosbag/2024-10-17-11-40-46.bag'
topic = '/velodyne_points'  # replace with your actual lidar topic
output_dir = '/media/idiot/T7 Shield/LiDAR Data/CP_lexus/lidar_5'
save_interval = 0.1                    # seconds, for 10 Hz

# === Prepare output directory ===
os.makedirs(output_dir, exist_ok=True)

# === Extraction Loop ===
last_time_sec = None

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[topic]):
        time_now_sec = t.to_sec()

        # Enforce 10Hz extraction (every 0.1s)
        if last_time_sec is None or (time_now_sec - last_time_sec >= save_interval):
            # Extract point cloud: x, y, z, intensity
            points = np.array(list(
                pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
            ))

            # Add global timestamp (in milliseconds) as a new column
            timestamp_ms = int(round(time_now_sec * 1000))
            timestamp_col = np.full((points.shape[0], 1), timestamp_ms)
            points_with_timestamp = np.hstack((points, timestamp_col))

            # Save to file
            np.save(os.path.join(output_dir, f"{timestamp_ms}.npy"), points_with_timestamp)
            print(f"Saved frame at {timestamp_ms} ms with shape {points_with_timestamp.shape}")

            last_time_sec = time_now_sec
