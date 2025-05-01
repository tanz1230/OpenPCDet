#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 04:49:06 2025

@author: idiot
"""
import os
import argparse
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

print("ROS dependencies imported successfully!")

def convert_rosbag_to_npy(bag_file, topic, output_folder, time_resolution=0.1):
    """
    Convert point cloud data from a rosbag to .npy files.
    
    Parameters:
      bag_file (str): Path to the input rosbag file.
      topic (str): The ROS topic from which to read PointCloud2 messages.
      output_folder (str): Directory where the .npy files will be saved.
      time_resolution (float): Minimum time interval between saved frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    bag = rosbag.Bag(bag_file, 'r')
    frame_count = 0
    last_saved_time = None  # Store the last saved timestamp

    for _, msg, t in bag.read_messages(topics=[topic]):
        current_time = t.to_sec()  # Convert ROS time to seconds
        
        # Save the first frame, then only save frames at least `time_resolution` seconds apart
        if last_saved_time is None or (current_time - last_saved_time >= time_resolution):
            # Extract x, y, z, and intensity (modify field_names if necessary)
            points_list = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            points_array = np.array(points_list, dtype=np.float32)
            
            # === Custom Transformations for OpenPCDet ===

            # 1. Transform Coordinate System:
            # (Perform any coordinate transformations if necessary)
            #
            # 2. (Optional) Adjust z-axis origin:
            # (Perform any adjustments if necessary)

            # 3. Normalize or set intensity:
            if points_array.shape[1] < 4:
                # If intensity is missing, add a column of zeros
                intensity = np.zeros((points_array.shape[0], 1), dtype=np.float32)
                points_array = np.concatenate([points_array, intensity], axis=1)
            else:
                # Normalize intensity to [0, 1] if maximum value is greater than 1.
                max_intensity = np.max(points_array[:, 3])
                if max_intensity > 1:
                    points_array[:, 3] /= max_intensity
            
            # === Append the timestamp as the fifth attribute ===
            # Create a column of the current timestamp (same for all points in this frame)
            timestamp_array = np.full((points_array.shape[0], 1), current_time, dtype=np.float32)
            # Concatenate to form an array with 5 attributes per point
            points_array = np.concatenate([points_array, timestamp_array], axis=1)
            
            # Create a filename with a 6-digit index (e.g., 000001.npy)
            output_filename = os.path.join(output_folder, f"{frame_count:06d}.npy")
            np.save(output_filename, points_array)

            print(f"Saved frame {frame_count} at time {current_time:.2f}s to {output_filename}")

            frame_count += 1
            last_saved_time = current_time  # Update last saved time

    bag.close()
    print("Conversion complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert point cloud data in a rosbag to .npy files for OpenPCDet."
    )
    parser.add_argument("--bag_file", type=str, required=True, help="Path to the rosbag file.")
    parser.add_argument("--topic", type=str, default="/ouster/points",
                        help="Topic name containing the PointCloud2 messages. Default: /ouster/points")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Directory to save the .npy files (one per point cloud frame).")
    
    args = parser.parse_args()
    convert_rosbag_to_npy(args.bag_file, args.topic, args.output_folder)
