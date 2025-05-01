#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 02:36:35 2025

@author: idiot
"""
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os

bag_file = '/media/idiot/T7 Shield/LiDAR Data/CP_lexus/10_17_2024_rosbag/2024-10-17-11-40-46.bag'
topic = '/velodyne_points'  # replace with your actual lidar topic
output_dir = '/media/idiot/T7 Shield/LiDAR Data/CP_lexus/lidar'

os.makedirs(output_dir, exist_ok=True)

last_time_sec = None
save_interval = 0.1  # 10Hz => 0.1 second

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[topic]):
        time_now_sec = t.to_sec()

        if last_time_sec is None or (time_now_sec - last_time_sec >= save_interval):
            points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))
            timestamp_ms = int(round(time_now_sec * 1000))  # rounded to milliseconds
            np.save(os.path.join(output_dir, f"{timestamp_ms}.npy"), points)
            print(f"Saved frame at {timestamp_ms} ms")
            last_time_sec = time_now_sec