"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 0, 0],    # label 0
    [0, 1, 0],    # label 1
    [0, 0, 1],    # label 2
    [1, 1, 0],    # label 3
    [1, 0, 1],    # label 4
    [0, 1, 1],    # label 5
    [0.5, 0.5, 0],  # label 6
    [0.5, 0, 0.5],  # label 7
    [0, 0.5, 0.5],  # label 8
    [0.5, 0.5, 0.5], # label 9
    [1, 0.5, 0]   # label 10
]
#extend color map


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None,
                ref_scores=None, point_colors=None, draw_origin=True,
                save_path=None, win_w=1920, win_h=1080):

    # ---------- 1. Tensor → NumPy (float32) -------------------------------
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    points = np.asarray(points, dtype=np.float32)       # <- key line

    # ---------- 2. Create window ------------------------------------------
    vis = open3d.visualization.Visualizer()
    vis.create_window("OpenPCDet‑Open3D", win_w, win_h)
    ...

    ro = vis.get_render_option()
    ro.point_size = 1.0
    ro.background_color = np.zeros(3)

    # -------- draw origin ----------------------------------------------------
    if draw_origin:
        vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(
                         size=1.0, origin=[0, 0, 0]))

    # -------- point cloud ----------------------------------------------------
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[:, :3])
    if point_colors is None:
        pcd.colors = open3d.utility.Vector3dVector(
                        np.ones((points.shape[0], 3)))
    else:
        pcd.colors = open3d.utility.Vector3dVector(point_colors)
    vis.add_geometry(pcd)

    # -------- bounding boxes -------------------------------------------------
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    # -------------------------------------------------------------------------
    # Camera: orthographic bird‑eye.  'front' points from look‑at to camera.
    # Put the camera at (0,0,+Z) looking straight down (front=[0,0,1]).
    # -------------------------------------------------------------------------
    vc = vis.get_view_control()
    vc.set_lookat([0, 0, 0])
    vc.set_front([0, 0, -1])          # camera is above, looking down
    vc.set_up([0, 1, 0])            # +Y points to the top of image
    vc.set_zoom(0.45)                # smaller → zoom‑in, bigger → zoom‑out
    vc.set_constant_z_far(500.0)     # avoid far‑plane clipping


    # ensure the visualiser has one redraw cycle before capture
    for _ in range(2):
        vis.poll_events()
        vis.update_renderer()

    # -------- interactive or head‑less ---------------------------------------
    if save_path is None:
        vis.run()                    # interactive window
    else:
        vis.capture_screen_image(save_path)

    vis.destroy_window()




def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
