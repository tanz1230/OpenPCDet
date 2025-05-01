import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader_CP
from pcdet.models import build_network
from pcdet.utils import common_utils



def parse_config():
    parser = argparse.ArgumentParser(description='Evaluation of two cooperative perception models')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='Specify the config YAML file for the first model')
    parser.add_argument('--ckpt_1', type=str, required=True,
                        help='Directory containing first model checkpoints')
    parser.add_argument('--ckpt_2', type=str, required=True,
                        help='Directory containing second model checkpoints')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--infer_time', action='store_true', default=False,
                        help='Measure inference latency')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--dist_th', type=int, default=140, help='lidar detection range')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--generate_predictions', action='store_true', default=False, help='whether to generate predictions for two vehicels')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    # Load config
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    # Seed for reproducibility
    np.random.seed(1024)

    return args, cfg

def generate_single_predictions(model, test_loader, args, ckpt, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    
    # start evaluation
    eval_utils.single_prediction_CP(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )

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

def main():
    
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #Create Output Dir 1
    eval_output_dir_1 = output_dir / 'eval_model_1'
    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt_1) if args.ckpt_1 is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir_1 = eval_output_dir_1 / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG_1.DATA_SPLIT['test']
    else:
        eval_output_dir_1 = eval_output_dir_1 / 'eval_all_default'
    if args.eval_tag is not None:
        eval_output_dir_1 = eval_output_dir_1 / args.eval_tag   
    eval_output_dir_1.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir_1 / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    #Create Output Dir 2
    eval_output_dir_2 = output_dir / 'eval_model_2'
    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt_1) if args.ckpt_1 is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir_2 = eval_output_dir_2 / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG_2.DATA_SPLIT['test']
    else:
        eval_output_dir_2 = eval_output_dir_2 / 'eval_all_default'
    if args.eval_tag is not None:
        eval_output_dir_2 = eval_output_dir_2 / args.eval_tag   
    eval_output_dir_2.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir_2 / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    
    
    pose_cache_1 = build_pose_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/gps_astuff')
    pose_cache_2 = build_pose_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/gps_tesla')
    
    extrinsic_cache_1 = build_extrinsic_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/tf_astuff', inverse=False) # True if IMU to Sensor
    extrinsic_cache_2 = build_extrinsic_cache('/media/idiot/OS/Project/Cooperative Perception/V2V4Real/tf_tesla', inverse=False) # False if Sensor to IMU

    test_set_1, test_loader_1, sampler_1 = build_dataloader_CP(
        dataset_cfg=cfg.DATA_CONFIG_1,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    
    if args.generate_predictions: # the two dataloader cannot be built at the same time
        test_set_2, test_loader_2, sampler_2 = build_dataloader_CP(
            dataset_cfg=cfg.DATA_CONFIG_2,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
    
        model_1 = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set_1)
        model_2 = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set_2)
        print("#############model built successfully")
    
        # Generate Prediction Results for model 1
        with torch.no_grad(): # generate prediction results
            generate_single_predictions(model_1, test_loader_1, args, args.ckpt_1, eval_output_dir_1, logger, epoch_id, dist_test=dist_test)
        print("############# Vehicle 1 Prediction Results Generated Successfully")
        
        # Generate Prediction Results for model 2
        with torch.no_grad():
            generate_single_predictions(model_2, test_loader_2, args, args.ckpt_2, eval_output_dir_2, logger, epoch_id, dist_test=dist_test)
        print("############# Vehicle 2 Prediction Results Generated Successfully")
            
    result_dir=output_dir / 'CP_final'
    eval_utils.eval_CP(cfg, args, logger, eval_output_dir_1, eval_output_dir_2, 
                          test_loader_1,
                          pose_cache_1=pose_cache_1, pose_cache_2=pose_cache_2, 
                          extrinsic_cache_1=extrinsic_cache_1, 
                          extrinsic_cache_2=extrinsic_cache_2, 
                          dist_th=args.dist_th,
                          dist_test=dist_test, result_dir=result_dir)


if __name__ == '__main__':
    main()

