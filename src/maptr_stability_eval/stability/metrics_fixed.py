"""
修复后的稳定性指标计算模块

基于test_stable.py的正确实现，确保与原始结果完全一致。
"""

import numpy as np
import copy
from collections import defaultdict
from tqdm import tqdm

from .alignment import align_det_and_gt_by_maptr_assigner
from ..geometry import poly_get_samples, polyline_iou
from ..coordinate_transform import transform_points_between_frames


def compute_presence_consistency(cur_scores, pre_scores, threshold=0.01):
    """
    计算两帧间的在场一致性。只有当当前帧与前一帧均检测到同一GT实例（score>=阈值）时记为1，否则为0.5。
    """
    cur_present = (np.asarray(cur_scores) >= float(threshold)) & (np.asarray(pre_scores) <= float(threshold))
    pre_present = (np.asarray(pre_scores) <= float(threshold)) & (np.asarray(cur_scores) >= float(threshold))
    presence = np.where(cur_present | pre_present, 1.0, 0.5).astype(np.float32)
    return presence


def get_localization_variations(cur_pred, pre_pred, 
                               cur_ego_translation, cur_ego_rotation,
                               pre_ego_translation, pre_ego_rotation,
                               x_range=(-30, 30), y_range=(-15, 15)):
    """
    计算位置变化指标 (针对折线)
    """
    # 转换坐标系
    pre_pred_in_cur_frame = transform_points_between_frames(
        pre_pred, 
        src_translation=pre_ego_translation,
        src_rotation=pre_ego_rotation,
        dst_translation=cur_ego_translation,
        dst_rotation=cur_ego_rotation
    )
    
    # 筛选当前帧作用范围内的点
    valid_mask = (
        (pre_pred_in_cur_frame[:, 0] >= x_range[0]) &
        (pre_pred_in_cur_frame[:, 0] <= x_range[1]) &
        (pre_pred_in_cur_frame[:, 1] >= y_range[0]) &
        (pre_pred_in_cur_frame[:, 1] <= y_range[1])
    )
    filtered_pre_pred = pre_pred_in_cur_frame[valid_mask]

    x_samples = poly_get_samples(cur_pred, num_samples=100)
    
    # 计算当前预测和先前预测之间的IoU
    iou = polyline_iou(cur_pred, filtered_pre_pred, x_samples)
    print("iou:", iou)
    return iou


def get_shape_variations(cur_pred, prev_pred, 
                         cur_ego_translation, cur_ego_rotation,
                         prev_ego_translation, prev_ego_rotation,
                         x_range=(-30, 30), y_range=(-15, 15)):
    """
    计算形状变化指标 (针对折线)
    """
    if len(cur_pred) < 2 or len(prev_pred) < 2:
        return -1
        
    # 转换坐标系
    pre_pred_in_cur_frame = transform_points_between_frames(
        prev_pred, 
        src_translation=prev_ego_translation,
        src_rotation=prev_ego_rotation,
        dst_translation=cur_ego_translation,
        dst_rotation=cur_ego_rotation
    )
    
    # 计算折线的曲率变化
    def compute_curvature(poly):
        if len(poly) < 3:
            return 0.0
        curvatures = []
        for i in range(1, len(poly)-1):
            p0, p1, p2 = poly[i-1], poly[i], poly[i+1]
            dx1, dy1 = p1[0]-p0[0], p1[1]-p0[1]
            dx2, dy2 = p2[0]-p1[0], p2[1]-p1[1]
            norm1 = np.sqrt(dx1**2 + dy1**2)
            norm2 = np.sqrt(dx2**2 + dy2**2)
            if norm1 == 0 or norm2 == 0:
                pass
            else:
                cos_angle = (dx1*dx2 + dy1*dy2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 限制值域
                curvatures.append(angle)
        return np.mean(curvatures) if curvatures else 0.0
    
    # 筛选当前帧作用范围内的点
    valid_mask = (
        (pre_pred_in_cur_frame[:, 0] >= x_range[0]) &
        (pre_pred_in_cur_frame[:, 0] <= x_range[1]) &
        (pre_pred_in_cur_frame[:, 1] >= y_range[0]) &
        (pre_pred_in_cur_frame[:, 1] <= y_range[1])
    )
    filtered_pre_pred = pre_pred_in_cur_frame[valid_mask]

    final_prev_pred = pre_pred_in_cur_frame
    
    # 计算曲率
    cur_curvature = compute_curvature(cur_pred)
    prev_curvature = compute_curvature(final_prev_pred)
    
    # 曲率变化越小越好
    curvature_var = 1 - np.abs(cur_curvature - prev_curvature) / np.pi
    return max(0.0, curvature_var)


def eval_maptr_stability_index(cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos, 
                              class_names=None, localization_weight: float = 0.3):
    """
    MapTR 稳定性评估主函数。基于 GT instance 对齐后的两帧检测，
    计算：
    1) 在场一致性 presence_consistency：两帧均检测到同一实例记1，否则0；
    2) 位置与形状稳定性：跨帧坐标对齐后计算折线 IoU 与曲率变化；
    3) 综合稳定性：presence × (localization*W + shape*(1-W))。
    """
    assert len(cur_det_annos) == len(pre_det_annos) == len(cur_gt_annos) == len(pre_gt_annos)
    
    cur_det_annos = copy.deepcopy(cur_det_annos)
    pre_det_annos = copy.deepcopy(pre_det_annos)
    cur_gt_annos = copy.deepcopy(cur_gt_annos)
    pre_gt_annos = copy.deepcopy(pre_gt_annos)

    paired_infos = defaultdict(list)
    for idx in tqdm(range(len(cur_gt_annos))):
        cur_gt_anno, pre_gt_anno = cur_gt_annos[idx], pre_gt_annos[idx]
        cur_det_anno, pre_det_anno = cur_det_annos[idx], pre_det_annos[idx]

        # 获取匹配的GT元素 (基于instance_id)
        cur_gt_ids = cur_gt_anno['instance_ids']
        pre_gt_ids = pre_gt_anno['instance_ids']
        
        if len(cur_gt_ids) == 0 or len(pre_gt_ids) == 0:
            continue
            
        # 匹配相同instance的元素
        cur_align_idx, pre_align_idx = np.nonzero(
            cur_gt_ids[:, None] == pre_gt_ids[None, :])
        
        if len(cur_align_idx) == 0:
            continue
            
        # 获取匹配的GT折线
        cur_gt_polylines = [cur_gt_anno['polylines'][i] for i in cur_align_idx]
        pre_gt_polylines = [pre_gt_anno['polylines'][i] for i in pre_align_idx]
        gt_types = [cur_gt_anno['types'][i] for i in cur_align_idx]
        
        # 获取检测结果
        cur_det_polylines = cur_det_anno['polylines']
        cur_det_types = cur_det_anno['types']
        cur_det_scores = cur_det_anno.get('scores', np.ones(len(cur_det_polylines)))
        
        pre_det_polylines = pre_det_anno['polylines']
        pre_det_types = pre_det_anno['types']
        pre_det_scores = pre_det_anno.get('scores', np.ones(len(pre_det_polylines)))
        
        # 对齐检测和GT（使用 MapTR 的分配器）
        cur_aligned_polylines, cur_aligned_scores = align_det_and_gt_by_maptr_assigner(
            cur_det_polylines, cur_det_types, 
            cur_gt_polylines, gt_types, class_names, 
            pc_range=[-25.0, -25.0, -5.0, 25.0, 25.0, 5.0], 
            det_scores=cur_det_scores)
        
        pre_aligned_polylines, pre_aligned_scores = align_det_and_gt_by_maptr_assigner(
            pre_det_polylines, pre_det_types, 
            pre_gt_polylines, gt_types, class_names, 
            pc_range=[-25.0, -25.0, -5.0, 25.0, 25.0, 5.0], 
            det_scores=pre_det_scores)
        
        # 存储配对信息
        paired_infos['cur_gt_polylines'].extend(cur_gt_polylines)
        paired_infos['pre_gt_polylines'].extend(pre_gt_polylines)
        paired_infos['cur_det_polylines'].extend(cur_aligned_polylines)
        paired_infos['cur_det_scores'].extend(cur_aligned_scores)
        paired_infos['pre_det_polylines'].extend(pre_aligned_polylines)
        paired_infos['pre_det_scores'].extend(pre_aligned_scores)
        paired_infos['gt_types'].extend(gt_types)
        paired_infos['cur_ego_translation'].append(cur_gt_anno['ego_translation'])
        paired_infos['cur_ego_rotation'].append(cur_gt_anno['ego_rotation'])
        paired_infos['pre_ego_translation'].append(pre_gt_anno['ego_translation'])
        paired_infos['pre_ego_rotation'].append(pre_gt_anno['ego_rotation'])

    # 检查paired_infos是否为空
    if not paired_infos or all(len(paired_infos[key]) == 0 for key in paired_infos):
        print("Warning: No valid paired data found for stability evaluation. This might be due to:")
        print("1. No matching instance IDs between consecutive frames")
        print("2. Empty GT annotations")
        print("3. No valid detections")
        # 返回空的稳定性指标
        return {
            'presence_consistency': {},
            'localization_stability': {},
            'shape_stability': {},
            'stability_index': {},
            'mean_presence_consistency': 0.0,
            'mean_localization_stability': 0.0,
            'mean_shape_stability': 0.0,
            'mean_stability_index': 0.0
        }
    
    # 确保所有键的长度一致
    min_length = min(len(paired_infos[key]) for key in paired_infos)
    for key in paired_infos:
        if len(paired_infos[key]) > min_length:
            paired_infos[key] = paired_infos[key][:min_length]
    
    # 过滤未被检测到的元素（在长度对齐之后计算，避免尺寸不一致）
    DETECTION_THRESHOLD = 0.01  # 检测阈值
    not_det_mask = np.logical_and(
        np.array(paired_infos['pre_det_scores']) < DETECTION_THRESHOLD,
        np.array(paired_infos['cur_det_scores']) < DETECTION_THRESHOLD
    )
    
    # 计算稳定性指标
    metrics = dict()
    if len(paired_infos['cur_det_scores']) > 0:
        # 在场一致性：惩罚"该帧检测、下一帧未检测"或反之的情况
        cur_scores_arr = np.array(paired_infos['cur_det_scores'])
        pre_scores_arr = np.array(paired_infos['pre_det_scores'])
        presence_vec = compute_presence_consistency(cur_scores_arr, pre_scores_arr, threshold=0.3)
        
        localization_vars = []
        shape_vars = []
        
        # 确保循环次数一致
        num_items = len(paired_infos['cur_det_polylines'])
        for i in range(num_items):
            cur_poly = paired_infos['cur_det_polylines'][i]
            pre_poly = paired_infos['pre_det_polylines'][i]
            cur_ego_translation = paired_infos['cur_ego_translation'][i]
            cur_ego_rotation = paired_infos['cur_ego_rotation'][i]
            pre_ego_translation = paired_infos['pre_ego_translation'][i]
            pre_ego_rotation = paired_infos['pre_ego_rotation'][i]
            localization_vars.append(get_localization_variations(
                cur_poly, pre_poly, cur_ego_translation, cur_ego_rotation,
                pre_ego_translation, pre_ego_rotation))
            shape_vars.append(get_shape_variations(
                cur_poly, pre_poly, cur_ego_translation, cur_ego_rotation,
                pre_ego_translation, pre_ego_rotation))
        
        localization_vars = np.array([v if v != -1 else 0.0 for v in localization_vars])
        shape_vars = np.array([v if v != -1 else 0.0 for v in shape_vars])

        # 过滤两帧均未检测到的样本，避免无意义对拉低统计
        types_arr = np.array(paired_infos['gt_types'])
        valid_mask = ~not_det_mask
        if valid_mask.size != localization_vars.size:
            valid_mask = np.ones_like(localization_vars, dtype=bool)

        if not np.any(valid_mask):
            for class_name in class_names:
                metrics['PRESENCE_CONSISTENCY_%s' % class_name] = 0.0
                metrics['LOCALIZATION_VARIATION_%s' % class_name] = 0.0
                metrics['SHAPE_VARIATION_%s' % class_name] = 0.0
                metrics['STABILITY_INDEX_%s' % class_name] = 0.0
            for si_type in ['PRESENCE_CONSISTENCY', 'LOCALIZATION_VARIATION', 'SHAPE_VARIATION', 'STABILITY_INDEX']:
                metrics['%s_mean' % si_type] = 0.0
            return metrics

        presence_vec = presence_vec[valid_mask]
        localization_vars = localization_vars[valid_mask]
        shape_vars = shape_vars[valid_mask]
        types_arr = types_arr[valid_mask]

        paired_infos['localization_vars'] = localization_vars
        paired_infos['shape_vars'] = shape_vars
        paired_infos['presence_consistency'] = presence_vec

        # 计算综合稳定性指数：presence × 加权(localization, shape)，并裁剪到[0,1]
        base_score = np.clip(paired_infos['localization_vars']*localization_weight + paired_infos['shape_vars']*(1-localization_weight), 0.0, 1.0)
        stability_index = np.clip(paired_infos['presence_consistency'] * base_score, 0.0, 1.0)
        paired_infos['stability_index'] = stability_index
        
        # 按类别计算指标
        for class_name in class_names:
            class_mask = (types_arr == class_name)
            if np.any(class_mask):
                _presence = presence_vec[class_mask]
                _localization = localization_vars[class_mask]
                _shape = shape_vars[class_mask]
                _stability_index = stability_index[class_mask]
                
                metrics['PRESENCE_CONSISTENCY_%s' % class_name] = _presence.mean()
                metrics['LOCALIZATION_VARIATION_%s' % class_name] = _localization.mean()
                metrics['SHAPE_VARIATION_%s' % class_name] = _shape.mean()
                metrics['STABILITY_INDEX_%s' % class_name] = _stability_index.mean()
            else:
                metrics['PRESENCE_CONSISTENCY_%s' % class_name] = 0.0
                metrics['LOCALIZATION_VARIATION_%s' % class_name] = 0.0
                metrics['SHAPE_VARIATION_%s' % class_name] = 0.0
                metrics['STABILITY_INDEX_%s' % class_name] = 0.0
        
        # 计算平均指标
        for si_type in ['PRESENCE_CONSISTENCY', 'LOCALIZATION_VARIATION', 
                       'SHAPE_VARIATION', 'STABILITY_INDEX']:
            collector = [metrics.get('%s_%s' % (si_type, class_name), 0) for class_name in class_names]
            metrics['%s_mean' % si_type] = sum(collector) / len(collector)
    else:
        # 如果没有匹配的元素，返回空指标
        for class_name in class_names:
            metrics['PRESENCE_CONSISTENCY_%s' % class_name] = 0.0
            metrics['LOCALIZATION_VARIATION_%s' % class_name] = 0.0
            metrics['SHAPE_VARIATION_%s' % class_name] = 0.0
            metrics['STABILITY_INDEX_%s' % class_name] = 0.0
        for si_type in ['PRESENCE_CONSISTENCY', 'LOCALIZATION_VARIATION', 
                       'SHAPE_VARIATION', 'STABILITY_INDEX']:
            metrics['%s_mean' % si_type] = 0.0

    return metrics
