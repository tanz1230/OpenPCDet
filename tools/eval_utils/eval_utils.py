import pickle
import time
import os

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu

from copy import deepcopy

def geodetic_to_ecef(lat, lon, h):
    """
    Convert geodetic (deg,deg,m) → ECEF (m).
    WGS‑84 ellipsoid.
    """
    # WGS‑84 parameters
    a  = 6378137.0
    e2 = 6.69437999014e-3

    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    N   = a / np.sqrt(1 - e2 * np.sin(phi)**2)

    X = (N + h) * np.cos(phi) * np.cos(lam)
    Y = (N + h) * np.cos(phi) * np.sin(lam)
    Z = (N * (1 - e2) + h) * np.sin(phi)
    return np.array([X, Y, Z])

def euler_to_rot_matrix(roll, pitch, yaw):
    """
    Build 3×3 rotation from roll, pitch, yaw (in radians).
    Uses R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1, 0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    return Rz @ Ry @ Rx

def enu_to_ecef_matrix(lat, lon):
    """
    Build 3×3 matrix mapping local ENU → ECEF at (lat, lon).
    """
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_lam, cos_lam = np.sin(lam), np.cos(lam)

    return np.array([
        [-sin_lam,            cos_lam,           0],
        [-sin_phi*cos_lam, -sin_phi*sin_lam,  cos_phi],
        [ cos_phi*cos_lam,  cos_phi*sin_lam,  sin_phi]
    ])

def make_body_to_ecef(lat, lon, alt, roll, pitch, yaw):
    """ Return a 3×3 R and 3×1 t so that p_ecef = R @ p_body + t. """
    # Rotation: body→ENU→ECEF
    R_att      = euler_to_rot_matrix(roll, pitch, yaw)        # body→ENU
    R_enu_ecef = enu_to_ecef_matrix(lat, lon)                 # ENU→ECEF
    R_body2ecef = R_enu_ecef @ R_att

    # Translation: vehicle origin in ECEF
    t_body2ecef = geodetic_to_ecef(lat, lon, alt)
    return R_body2ecef, t_body2ecef

def make_sensor_to_body(Tr_sensor_to_imu):
    Tr_sensor_to_imu = np.eye(4) ################################################################test, imu and sensor are the same
    """ Split your 4×4 into R and t for body←sensor. """
    R_s2b = Tr_sensor_to_imu[:3,:3]
    t_s2b = Tr_sensor_to_imu[:3, 3]
    return R_s2b, t_s2b

def make_body_to_sensor(R_s2b, t_s2b):
    """ Invert sensor→body to get body→sensor. """
    # R_s2b: 3×3 rotation from sensor into body
    # t_s2b: 3×1 translation from sensor origin to body origin, expressed in body frame
    # The inverse rotation is the transpose:
    R_b2s = R_s2b.T
    # The inverse translation is −Rᵀ * t
    t_b2s = -R_b2s @ t_s2b
    return R_b2s, t_b2s

def make_homogeneous(R, t):
    """ Build a 4×4 from R (3×3) and t (3,). """
    T = np.eye(4, dtype=R.dtype)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def transform_boxes_remote_to_local(boxes_remote,
                                    Tr_remote_to_imu,
                                    Tr_local_to_imu,
                                    lat,lon,alt, 
                                    roll,pitch,yaw,
                                    ref_lat,ref_lon,ref_alt,
                                    ref_roll,ref_pitch,ref_yaw):
        
    #If torch.Tensor, extract device/dtype and convert to NumPy ---
    is_torch = isinstance(boxes_remote, torch.Tensor)
    if is_torch:
        device = boxes_remote.device
        dtype  = boxes_remote.dtype
        # bring to CPU numpy
        boxes_remote = boxes_remote.detach().cpu().numpy()
    else:
        device = None
        dtype  = None
        
    # centers:
    N = boxes_remote.shape[0]
    centers = boxes_remote[:,:3]
    
    # generate T_remote2local
    #  A) Remote sensor → body → ECEF
    R_s2b, t_s2b = make_sensor_to_body(Tr_remote_to_imu)
    R_b2e, t_b2e = make_body_to_ecef(
        lat,lon,alt,
        roll,pitch,yaw
    )
    T_remote2ecef = make_homogeneous(R_b2e @ R_s2b,       # rotation: sensor→ECEF
                                     R_b2e @ t_s2b + t_b2e)  # translation
    
    #  B) ECEF → local body → local sensor
    # Invert the same pipeline for your vehicle:
    R_body2ecef_loc, t_body2ecef_loc = make_body_to_ecef(
        ref_lat,ref_lon,ref_alt,
        ref_roll,ref_pitch,ref_yaw
    )
    # body→sensor:
    R_b2s_loc, t_b2s_loc = make_body_to_sensor(*make_sensor_to_body(Tr_local_to_imu))
    # ECEF→body = inverse of body→ECEF
    R_e2b_loc = R_body2ecef_loc.T
    t_e2b_loc = -R_body2ecef_loc.T @ t_body2ecef_loc
    
    T_ecef2local = make_homogeneous(R_b2s_loc @ R_e2b_loc,
                                    R_b2s_loc @ t_e2b_loc + t_b2s_loc)
    
    #  C) Compose once:
    T_remote2local = T_ecef2local @ T_remote2ecef
    

    # homogeneous:
    homo = np.concatenate([centers, np.ones((N,1))], axis=1)  # (N,4)
    centers_loc = (T_remote2local @ homo.T).T[:, :3]

    # yaw: rotate each forward‐vector
    yaws = boxes_remote[:,6]
    fwd = np.stack([np.cos(yaws), np.sin(yaws), np.zeros_like(yaws)], axis=1)  # (N,3)
    # apply rotation part only:
    R = T_remote2local[:3,:3]
    fwd_loc = (R @ fwd.T).T
    new_yaw = np.arctan2(fwd_loc[:,1], fwd_loc[:,0])

    # assemble:
    boxes_local = boxes_remote.copy()
    boxes_local[:,:3] = centers_loc
    boxes_local[:, 6] = new_yaw
    
    # Convert back to torch if needed
    if is_torch:
        boxes_local = torch.from_numpy(boxes_local)
        boxes_local = boxes_local.to(device=device, dtype=dtype)

    return boxes_local

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def single_prediction_CP(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0


    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    
    pred_bank = {}
    for i, batch_dict in enumerate(dataloader):
        ts = batch_dict['frame_id'][0] #timestamp used to match the lidar frames from two vehicles
        
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        
        pred_bank[ts] = pred_dicts
        
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_bank, f)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Prediction Results Generation Done.*****************')


def eval_CP(cfg, args, logger, eval_output_dir_1, eval_output_dir_2, 
                      dataloader_1,
                      pose_cache_1=None, pose_cache_2=None, extrinsic_cache_1=None, extrinsic_cache_2=None, dist_th=None,
                      dist_test=False, result_dir=None):

    def save_to_kitti_label(prediction_output, save_dir, filename=None):
        """
        Args:
            prediction_output (dict): Single output as shown.
            save_dir (str): Directory to save the txt file.
            filename (str, optional): Custom filename. If None, use frame_id + .txt
        """
        name_array = prediction_output['name'].squeeze(axis=1)
        boxes_lidar = prediction_output['boxes_lidar'].squeeze(1)
        frame_id = prediction_output['frame_id']
    
        # Make sure save dir exists
        os.makedirs(save_dir, exist_ok=True)
    
        # Create filename if not provided
        if filename is None:
            filename = f"{frame_id}.txt"
    
        file_path = os.path.join(save_dir, filename)
    
        with open(file_path, 'w') as f:
            for i in range(len(boxes_lidar)):
                x, y, z, l, w, h, yaw = boxes_lidar[i]
                label = name_array[i]
                # Format: x y z l w h yaw label (score is not included in your example)
                f.write(f"{x:.2f} {y:.2f} {z:.2f} {l:.2f} {w:.2f} {h:.2f} {yaw:.2f} {label}\n")
    
    def weighted_boxes_fusion(
        preds_a, preds_b,
        alpha=0.6,          # weight on model‑A scores, (1‑alpha) on model‑B
        iou_thr=0.5,       # 3‑D IoU used in GPU NMS
        score_thr=0.05      # prune before NMS to save memory/time
    ):
        """
        Per‑frame late fusion:
          ‑ re‑weights scores,
          ‑ concatenates boxes,
          ‑ applies GPU NMS, returns OpenPCDet‑style list[dict].
        Args
        ----
        preds_a / preds_b : list[dict]  # output of each model for **one batch**
        Returns
        -------
        fused_preds : list[dict]        # same list length as input batch
        """
        fused_batch = []
        #print(preds_a)
        #print(preds_b)
    
        for pa, pb in zip(preds_a, preds_b):
            boxes  = torch.cat([pa['pred_boxes'], pb['pred_boxes']], dim=0)        # (M+N, 7)
            scores = torch.cat([pa['pred_scores']*alpha, pb['pred_scores']*(1-alpha)], dim=0)  # (M+N,)
            labels = torch.cat([pa['pred_labels'], pb['pred_labels']], dim=0)        # (M+N,)
    
            # optional score filter
            keep_conf = scores > score_thr
            boxes, scores, labels = boxes[keep_conf], scores[keep_conf], labels[keep_conf]
    
            # GPU NMS (per‑class NMS could be done by looping over labels)
            keep_idx = nms_gpu(boxes, scores, iou_thr)
    
            fused_batch.append({
                'pred_boxes': boxes[keep_idx],
                'pred_scores':      scores[keep_idx],
                'pred_labels': labels[keep_idx]
            })
    
        return fused_batch
    
    result_dir.mkdir(parents=True, exist_ok=True)
    label_dir = result_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset_1 = dataloader_1.dataset
    #dataset_2 = dataloader_2.dataset
    
    class_names = dataset_1.class_names #Two datasets share the same class names
    
    with open(eval_output_dir_1 / 'result.pkl', 'rb') as f:
        pred_bank_1 = pickle.load(f)
        
    with open(eval_output_dir_2 / 'result.pkl', 'rb') as f:
        pred_bank_2 = pickle.load(f)
    
    
    det_annos = []
    metric = {'gt_num': 0}
    for thr in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric[f'recall_roi_{thr}']  = 0
        metric[f'recall_rcnn_{thr}'] = 0
    
    flag = False
    for ts, preds_1 in pred_bank_1.items():
        if ts not in pred_bank_2:
            logger.warning(f'missing sensor-2 frame {ts}, skipping'); continue
        
        if ts == '004281':
            flag = True
        
        preds_2 = pred_bank_2[ts]
    
        # ---- look-up pose & extrinsic once per ts -----------------
        lat1, lon1, alt1, yaw1, pitch1, roll1 = pose_cache_1[ts]
        lat2, lon2, alt2, yaw2, pitch2, roll2 = pose_cache_2[ts]
        Tr_lidar1_to_imu = extrinsic_cache_1[ts]
        Tr_lidar2_to_imu = extrinsic_cache_2[ts]
        ref_lat, ref_lon, ref_alt = lat1, lon1, alt1
        ref_roll,ref_pitch,ref_yaw = roll1, pitch1, yaw1
        
        if flag:
            print(preds_2[0]['pred_boxes'])
        pred_boxes_2_T = transform_boxes_remote_to_local(preds_2[0]['pred_boxes'],
                                                    Tr_lidar2_to_imu,
                                                    Tr_lidar1_to_imu,
                                                    lat2,lon2,alt2, 
                                                    roll2,pitch2,yaw2,
                                                    ref_lat, ref_lon, ref_alt,
                                                    ref_roll,ref_pitch,ref_yaw)
        if flag:
            print(pred_boxes_2_T)
        
        preds_2[0]['pred_boxes'] = pred_boxes_2_T
    
        # ---- fuse -------------------------------------------------
        fused = weighted_boxes_fusion(preds_1, preds_2)
        if flag:
            print(fused)
            
        # ---- back to sensor-1 frame + range filter ----------------
        pred_dicts = []
        for d in fused:
            boxes = d['pred_boxes']
            
            if boxes.size(0) > 0:
                # compute 2-D distance on GPU
                dists = torch.norm(boxes[..., :2], p=2, dim=-1)  # shape: (N, 1)
                keep = (dists[:, 0] <= args.dist_th)             # shape: (N,)
                boxes  = boxes[keep]
                scores = d['pred_scores'][keep]
                labels = d['pred_labels'][keep]

            if boxes.size(0) == 0:
                continue
        
            d2 = d.copy()
            d2['pred_boxes'] = boxes
            d2['pred_scores'] = scores
            d2['pred_labels'] = labels
            pred_dicts.append(d2)
        
        if flag:
            print(pred_dicts)
        # ---- accumulate stats & annos -----------------------------
        #statistics_info(cfg, ret_bank_1[ts], metric, {})
        annos = dataset_1.generate_prediction_dicts(
                    {'frame_id':[ts]}, pred_dicts, class_names,
                    output_path=None)
        if flag:
            print(annos)

        save_to_kitti_label(annos[0], label_dir)#save labels for debugging
        flag = False
        det_annos += annos

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    
    ret_dict = {}

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    
    # Flatten and force every anno['name'] to a 1-D numpy array of strings
    for anno in det_annos:
        for key, val in list(anno.items()):
            # only touch arrays / tensors
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy()
            elif isinstance(val, np.ndarray):
                arr = val
            else:
                continue  
            # if there's a length-1 axis, remove it
            # e.g. (N,1,7) -> (N,7); (1,N,4) -> (N,4); etc.
            if arr.ndim > 1 and 1 in arr.shape:
                arr = np.squeeze(arr)
            # now, ensure 2D arrays are actually shaped (num_boxes, feature_dim)
            # and 1D arrays are (num_boxes,)
            # (you can add more targeted checks here if desired)
    
            # put it back into anno as a pure numpy array
            anno[key] = arr

    result_str, result_dict = dataset_1.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

if __name__ == '__main__':
    pass
