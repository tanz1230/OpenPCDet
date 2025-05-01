#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 18:44:51 2025

@author: idiot
"""
import argparse
import glob
import copy
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            # Note: This reshape assumes a point format with 5 values (e.g., for Nuscenes)
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # First config file for the first dataset/model pair.
    parser.add_argument('--cfg_file1', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the first config file for demo')
    # Second config file for the second dataset/model pair.
    parser.add_argument('--cfg_file2', type=str, default='cfgs/kitti_models/another.yaml',
                        help='specify the second config file for demo')

    parser.add_argument('--data_path', type=str, default='demo_data1',
                        help='specify the first point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the first pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')
    # Additional arguments for the second dataset and model.
    parser.add_argument('--data_path2', type=str, default='demo_data2',
                        help='specify the second point cloud data file or directory')
    parser.add_argument('--ckpt2', type=str, default=None, help='specify the second pretrained model')

    args = parser.parse_args()

    # Load the first configuration into cfg1.
    cfg1 = copy.deepcopy(cfg)
    cfg_from_yaml_file(args.cfg_file1, cfg1)

    # Load the second configuration into cfg2.
    cfg2 = copy.deepcopy(cfg)
    cfg_from_yaml_file(args.cfg_file2, cfg2)

    return args, cfg1, cfg2


def main():
    args, cfg1, cfg2 = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet with Two Models-------------------------')

    # --------------------- First dataset and model ---------------------
    demo_dataset1 = DemoDataset(
        dataset_cfg=cfg1.DATA_CONFIG, class_names=cfg1.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'First dataset: Total number of samples: \t{len(demo_dataset1)}')

    model1 = build_network(model_cfg=cfg1.MODEL, num_class=len(cfg1.CLASS_NAMES), dataset=demo_dataset1)
    model1.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model1.cuda()
    model1.eval()

    # --------------------- Second dataset and model ---------------------
    demo_dataset2 = DemoDataset(
        dataset_cfg=cfg2.DATA_CONFIG, class_names=cfg2.CLASS_NAMES, training=False,
        root_path=Path(args.data_path2), ext=args.ext, logger=logger
    )
    logger.info(f'Second dataset: Total number of samples: \t{len(demo_dataset2)}')

    model2 = build_network(model_cfg=cfg2.MODEL, num_class=len(cfg2.CLASS_NAMES), dataset=demo_dataset2)
    model2.load_params_from_file(filename=args.ckpt2, logger=logger, to_cpu=True)
    model2.cuda()
    model2.eval()

    # --------------------- Inference and Combined Visualization ---------------------
    # For demonstration, we'll process the first sample from each dataset.
    with torch.no_grad():
        # Process first sample for the first dataset/model pair.
        data_dict1 = demo_dataset1[0]
        data_dict1 = demo_dataset1.collate_batch([data_dict1])
        load_data_to_gpu(data_dict1)
        pred_dicts1, _ = model1.forward(data_dict1)
    
        # Process first sample for the second dataset/model pair.
        data_dict2 = demo_dataset2[0]
        data_dict2 = demo_dataset2.collate_batch([data_dict2])
        load_data_to_gpu(data_dict2)
        pred_dicts2, _ = model2.forward(data_dict2)
    
        # Convert point tensors to NumPy arrays after moving them to CPU.
        points1 = data_dict1['points'].cpu().numpy()
        points2 = data_dict2['points'].cpu().numpy()
        combined_points = np.concatenate([points1, points2], axis=0)
    
        # Similarly, convert predictions if they are tensors (if necessary).
        boxes1 = pred_dicts1[0]['pred_boxes']
        boxes2 = pred_dicts2[0]['pred_boxes']
        if isinstance(boxes1, torch.Tensor):
            boxes1 = boxes1.cpu().numpy()
        if isinstance(boxes2, torch.Tensor):
            boxes2 = boxes2.cpu().numpy()
        combined_boxes = np.concatenate([boxes1, boxes2], axis=0)
    
        scores1 = pred_dicts1[0]['pred_scores']
        scores2 = pred_dicts2[0]['pred_scores']
        if isinstance(scores1, torch.Tensor):
            scores1 = scores1.cpu().numpy()
        if isinstance(scores2, torch.Tensor):
            scores2 = scores2.cpu().numpy()
        combined_scores = np.concatenate([scores1, scores2], axis=0)
    
        labels1 = pred_dicts1[0]['pred_labels']
        labels2 = pred_dicts2[0]['pred_labels']
        if isinstance(labels1, torch.Tensor):
            labels1 = labels1.cpu().numpy()
        if isinstance(labels2, torch.Tensor):
            labels2 = labels2.cpu().numpy()
        combined_labels = np.concatenate([labels1, labels2], axis=0)
    
        # Visualize combined point cloud and predictions.
        V.draw_scenes(
            points=combined_points[:, 1:],  # Visualization function expects points[:, 1:] format.
            ref_boxes=combined_boxes,
            ref_scores=combined_scores,
            ref_labels=combined_labels
        )
    
        if not OPEN3D_FLAG:
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

