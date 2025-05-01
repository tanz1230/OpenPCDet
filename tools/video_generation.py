
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.cm as cm

# === Configuration ===
npy_folder = '/media/idiot/T7 Shield/LiDAR Data/CP_lexus/lidar_visualize' # folder with .npy point clouds
output_video = "/media/idiot/T7 Shield/LiDAR Data/CP_lexus/lidar_video_colormap.avi"
video_fps = 10
frame_size = (800, 800)               # (width, height)
scale = 10                            # scaling factor for coordinate mapping
colormap = cm.get_cmap('plasma')      # change to 'viridis', 'inferno', etc.

# === Load .npy files ===
npy_files = sorted(glob(os.path.join(npy_folder, "*.npy")))

# === Setup video writer ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video, fourcc, video_fps, frame_size)

# === Visualization loop ===
for npy_file in tqdm(npy_files, desc="Generating video"):
    points = np.load(npy_file)  # shape: (N, 4) or (N, 5)
    
    x, y = points[:, 0], points[:, 1]
    intensity = points[:, 3]
    norm_intensity = np.clip(intensity / np.max(intensity + 1e-6), 0, 1)  # normalize

    u = (x * scale + frame_size[0] / 2).astype(np.int32)
    v = (y * scale + frame_size[1] / 2).astype(np.int32)

    # Apply colormap
    colors = (colormap(norm_intensity)[:, :3] * 255).astype(np.uint8)

    # Create a black background image
    img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

    # Draw each point
    for i in range(len(u)):
        if 0 <= u[i] < frame_size[0] and 0 <= v[i] < frame_size[1]:
            bgr = tuple(int(c) for c in colors[i][::-1])  # RGB to BGR
            img[v[i], u[i]] = bgr

    # Add ego vehicle center lines (crosshair)
    cv2.line(img, (frame_size[0] // 2, 0), (frame_size[0] // 2, frame_size[1]), (60, 60, 60), 1)
    cv2.line(img, (0, frame_size[1] // 2), (frame_size[0], frame_size[1] // 2), (60, 60, 60), 1)

    # Optional: show timestamp if column 4 or 5 exists
    if points.shape[1] >= 5:
        timestamp = int(points[0, 4])
        cv2.putText(img, f"{timestamp} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Write frame to video
    video_writer.write(img)

video_writer.release()
print(f"✅ Video saved to: {output_video}")
#%%
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

# === Configuration ===
npy_folder = '/media/idiot/T7 Shield/LiDAR Data/CP_lexus/lidar_visualize' # folder with .npy point clouds
output_video = "/media/idiot/T7 Shield/LiDAR Data/CP_lexus/lidar_video_intensity.avi"
video_fps = 10
frame_size = (1200, 1200)               # (width, height)
scale = 10                            # scaling factor for coordinates

# === Load and sort .npy files ===
npy_files = sorted(glob(os.path.join(npy_folder, "*.npy")))

# === OpenCV Video Writer ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video, fourcc, video_fps, frame_size)

# === Frame Generation Loop ===
for npy_file in tqdm(npy_files, desc="Generating video"):
    points = np.load(npy_file)  # shape: (N, 4) or (N, 5)

    x, y = points[:, 0], points[:, 1]
    intensity = points[:, 3]
    norm_intensity = np.clip(intensity / np.max(intensity + 1e-6), 0, 1)

    u = (x * scale + frame_size[0] / 2).astype(np.int32)
    v = (y * scale + frame_size[1] / 2).astype(np.int32)

    # Create blank image
    img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

    # Draw each point based on normalized intensity (blue → white)
    for i in range(len(u)):
        if 0 <= u[i] < frame_size[0] and 0 <= v[i] < frame_size[1]:
            val = int(norm_intensity[i] * 255)
            img[v[i], u[i]] = (val, val, 255)  # bluish tone, bright with high intensity

    # Draw center crosshairs
    cv2.line(img, (frame_size[0] // 2, 0), (frame_size[0] // 2, frame_size[1]), (50, 50, 50), 1)
    cv2.line(img, (0, frame_size[1] // 2), (frame_size[0], frame_size[1] // 2), (50, 50, 50), 1)

    # Optional: show frame timestamp if exists
    if points.shape[1] >= 5:
        timestamp = int(points[0, 4])
        cv2.putText(img, f"{timestamp} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    video_writer.write(img)

video_writer.release()
print(f"✅ Intensity-based LiDAR video saved as: {output_video}")
