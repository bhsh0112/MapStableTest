"""
稳定性评估指标计算

包含在场一致性、位置稳定性、形状稳定性等核心指标的计算函数。
"""

import numpy as np
import copy
from collections import defaultdict
from tqdm import tqdm

from ..geometry import (
    poly_get_samples,
    polyline_iou,
    transform_points_between_frames
)


def compute_presence_consistency(cur_scores, pre_scores, threshold=0.01):
    """
    计算两帧间的在场一致性。只有当当前帧与前一帧均检测到同一GT实例（score>=阈值）时记为1，否则为0.5。
    
    Args:
        cur_scores: 当前帧对齐后的分数数组
        pre_scores: 前一帧对齐后的分数数组
        threshold: 视为检测到的分数阈值
        
    Returns:
        presence: 长度与输入一致的0.5/1一致性向量
    """
    cur_scores = np.asarray(cur_scores, dtype=np.float32)
    pre_scores = np.asarray(pre_scores, dtype=np.float32)
    threshold = float(threshold)
    
    # 与test_stable.py保持完全一致的逻辑（即使逻辑看起来有问题）
    cur_present = (cur_scores >= threshold) & (pre_scores <= threshold)
    pre_present = (pre_scores <= threshold) & (cur_scores >= threshold)
    presence = np.where(cur_present | pre_present, 0.2, 1.0).astype(np.float32)
    return presence


def get_localization_variations(cur_pred, pre_pred, 
                               cur_ego_translation, cur_ego_rotation,
                               pre_ego_translation, pre_ego_rotation,
                               x_range=(-30, 30), y_range=(-15, 15)):
    """
    计算位置变化指标 (针对折线)
    
    Args:
        cur_pred: 当前帧预测折线
        pre_pred: 前一帧预测折线
        cur_ego_translation: 当前帧自车平移
        cur_ego_rotation: 当前帧自车旋转
        pre_ego_translation: 前一帧自车平移
        pre_ego_rotation: 前一帧自车旋转
        x_range: x方向范围
        y_range: y方向范围
        
    Returns:
        iou: 两帧预测的IoU值
    """
    # 如果ego_rotation为None，使用默认值（无旋转）
    # if cur_ego_rotation is None:
    #     from pyquaternion import Quaternion
    #     cur_ego_rotation = Quaternion()
    # if pre_ego_rotation is None:
    #     from pyquaternion import Quaternion
    #     pre_ego_rotation = Quaternion()
    
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
    # print("iou:", iou)
    return iou


def get_shape_variations(cur_pred, prev_pred, 
                         cur_ego_translation, cur_ego_rotation,
                         prev_ego_translation, prev_ego_rotation,
                         x_range=(-30, 30), y_range=(-15, 15)):
    """
    计算形状变化指标 (针对折线)
    
    Args:
        cur_pred: 当前帧预测折线
        prev_pred: 前一帧预测折线
        cur_ego_translation: 当前帧自车平移
        cur_ego_rotation: 当前帧自车旋转
        prev_ego_translation: 前一帧自车平移
        prev_ego_rotation: 前一帧自车旋转
        x_range: x方向范围
        y_range: y方向范围
        
    Returns:
        curvature_var: 曲率变化指标
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
    
    if len(cur_pred) < 3 or len(final_prev_pred) < 3:
        return 1
    else:
        cur_curvature = compute_curvature(cur_pred)
        prev_curvature = compute_curvature(pre_pred_in_cur_frame)
    
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
    
    Args:
        cur_det_annos: 当前帧预测注释（按场景时间排序后配对）
        pre_det_annos: 前一帧预测注释
        cur_gt_annos: 当前帧 GT 注释
        pre_gt_annos: 前一帧 GT 注释
        class_names: 评估类别列表
        localization_weight: W∈[0,1]，综合稳定性中位置项权重
        
    Returns:
        metrics: 指标字典，包含各类与均值的 presence/localization/shape/SI
    """
    # print(f"\n=== 稳定性指标计算开始 ===")
    # print(f"当前帧检测数量: {len(cur_det_annos)}")
    # print(f"前一帧检测数量: {len(pre_det_annos)}")
    # print(f"当前帧GT数量: {len(cur_gt_annos)}")
    # print(f"前一帧GT数量: {len(pre_gt_annos)}")
    # print(f"评估类别: {class_names}")
    # print(f"位置权重: {localization_weight}")
    
    assert len(cur_det_annos) == len(pre_det_annos) == len(cur_gt_annos) == len(pre_gt_annos)
    
    cur_det_annos = copy.deepcopy(cur_det_annos)
    pre_det_annos = copy.deepcopy(pre_det_annos)
    cur_gt_annos = copy.deepcopy(cur_gt_annos)
    pre_gt_annos = copy.deepcopy(pre_gt_annos)

    paired_infos = defaultdict(list)
    valid_pairs = 0
    
    print(f"\n=== 开始处理配对数据 ===")
    for idx in tqdm(range(len(cur_gt_annos))):
        cur_gt_anno, pre_gt_anno = cur_gt_annos[idx], pre_gt_annos[idx]
        cur_det_anno, pre_det_anno = cur_det_annos[idx], pre_det_annos[idx]
        
        # if idx < 3:  # 只对前3对数据打印详细信息
        #     print(f"\n--- 处理第 {idx+1} 对数据 ---")
        #     print(f"当前帧检测折线数: {len(cur_det_anno.get('polylines', []))}")
        #     print(f"前一帧检测折线数: {len(pre_det_anno.get('polylines', []))}")
        #     print(f"当前帧GT折线数: {len(cur_gt_anno.get('polylines', []))}")
        #     print(f"前一帧GT折线数: {len(pre_gt_anno.get('polylines', []))}")
        #     print(f"当前帧检测类型: {cur_det_anno.get('types', [])[:5]}...")
        #     print(f"前一帧检测类型: {pre_det_anno.get('types', [])[:5]}...")
        #     print(f"当前帧GT类型: {cur_gt_anno.get('types', [])[:5]}...")
        #     print(f"前一帧GT类型: {pre_gt_anno.get('types', [])[:5]}...")

        # 获取匹配的GT元素 (基于instance_id)
        cur_gt_ids = cur_gt_anno['instance_ids']
        pre_gt_ids = pre_gt_anno['instance_ids']
        
        # if idx < 3:
        #     print(f"当前帧GT instance_ids: {cur_gt_ids[:5]}...")
        #     print(f"前一帧GT instance_ids: {pre_gt_ids[:5]}...")
        
        if len(cur_gt_ids) == 0 or len(pre_gt_ids) == 0:
            if idx < 3:
                print(f"跳过第 {idx+1} 对数据：GT instance_ids为空")
            continue
            
        # 匹配相同instance的元素
        cur_align_idx, pre_align_idx = np.nonzero(
            cur_gt_ids[:, None] == pre_gt_ids[None, :])
        
        if idx < 3:
            print(f"匹配的instance数量: {len(cur_align_idx)}")
        
        if len(cur_align_idx) == 0:
            if idx < 3:
                print(f"跳过第 {idx+1} 对数据：没有匹配的instance")
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
        from .alignment import align_det_and_gt_by_maptr_assigner
        
        cur_aligned_polylines, cur_aligned_scores = align_det_and_gt_by_maptr_assigner(
            cur_det_polylines, cur_det_types, 
            cur_gt_polylines, gt_types, class_names, 
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            # pc_range=[-25.0, -25.0, -5.0, 25.0, 25.0, 5.0], 
            det_scores=cur_det_scores)
        
        pre_aligned_polylines, pre_aligned_scores = align_det_and_gt_by_maptr_assigner(
            pre_det_polylines, pre_det_types, 
            pre_gt_polylines, gt_types, class_names, 
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0], 
            # pc_range=[-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
            det_scores=pre_det_scores)
        
        # 存储配对信息
        paired_infos['cur_gt_polylines'].extend(cur_gt_polylines)
        paired_infos['pre_gt_polylines'].extend(pre_gt_polylines)
        paired_infos['cur_det_polylines'].extend(cur_aligned_polylines)
        paired_infos['cur_det_scores'].extend(cur_aligned_scores)
        paired_infos['pre_det_polylines'].extend(pre_aligned_polylines)
        paired_infos['pre_det_scores'].extend(pre_aligned_scores)
        
        # 直接扩展gt_types（与test_stable.py保持一致）
        paired_infos['gt_types'].extend(gt_types)
        
        # 存储ego pose信息（与test_stable.py保持一致）
        paired_infos['cur_ego_translation'].append(cur_gt_anno['ego_translation'])
        paired_infos['cur_ego_rotation'].append(cur_gt_anno['ego_rotation'])
        paired_infos['pre_ego_translation'].append(pre_gt_anno['ego_translation'])
        paired_infos['pre_ego_rotation'].append(pre_gt_anno['ego_rotation'])
        
        valid_pairs += 1

    # 检查paired_infos是否为空
    print(f"\n=== 配对数据统计 ===")
    print(f"有效配对数量: {valid_pairs}")
    for key in paired_infos:
        print(f"{key}: {len(paired_infos[key])}")
    
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
    print(f"\n=== 开始计算稳定性指标 ===")
    metrics = dict()
    if len(paired_infos['cur_det_scores']) > 0:
        # 在场一致性：惩罚"该帧检测、下一帧未检测"或反之的情况
        cur_scores_arr = np.array(paired_infos['cur_det_scores'])
        pre_scores_arr = np.array(paired_infos['pre_det_scores'])
        print(f"当前帧分数范围: [{cur_scores_arr.min():.4f}, {cur_scores_arr.max():.4f}]")
        print(f"前一帧分数范围: [{pre_scores_arr.min():.4f}, {pre_scores_arr.max():.4f}]")
        
        presence_vec = compute_presence_consistency(cur_scores_arr, pre_scores_arr, threshold=0.3)
        print(f"在场一致性向量长度: {len(presence_vec)}")
        print(f"在场一致性统计: 均值={presence_vec.mean():.4f}, 最小值={presence_vec.min():.4f}, 最大值={presence_vec.max():.4f}")
        
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
        print(f"\n=== 按类别计算指标 ===")
        print(f"总样本数: {len(types_arr)}")
        print(f"类别分布: {dict(zip(*np.unique(types_arr, return_counts=True)))}")
        
        for class_name in class_names:
            class_mask = (types_arr == class_name)
            class_count = np.sum(class_mask)
            print(f"\n处理类别: {class_name} (样本数: {class_count})")
            
            if np.any(class_mask):
                _presence = presence_vec[class_mask]
                _localization = localization_vars[class_mask]
                _shape = shape_vars[class_mask]
                _stability_index = stability_index[class_mask]
                
                print(f"  在场一致性: 均值={_presence.mean():.4f}, 范围=[{_presence.min():.4f}, {_presence.max():.4f}]")
                print(f"  位置变化: 均值={_localization.mean():.4f}, 范围=[{_localization.min():.4f}, {_localization.max():.4f}]")
                print(f"  形状变化: 均值={_shape.mean():.4f}, 范围=[{_shape.min():.4f}, {_shape.max():.4f}]")
                print(f"  稳定性指标: 均值={_stability_index.mean():.4f}, 范围=[{_stability_index.min():.4f}, {_stability_index.max():.4f}]")
                
                metrics['PRESENCE_CONSISTENCY_%s' % class_name] = _presence.mean()
                metrics['LOCALIZATION_VARIATION_%s' % class_name] = _localization.mean()
                metrics['SHAPE_VARIATION_%s' % class_name] = _shape.mean()
                metrics['STABILITY_INDEX_%s' % class_name] = _stability_index.mean()
            else:
                print(f"  类别 {class_name} 没有样本，设置为0")
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
