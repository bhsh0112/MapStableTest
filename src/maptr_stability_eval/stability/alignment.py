"""
检测与GT对齐模块

包含使用匈牙利算法和MapTR分配器进行检测结果与GT对齐的功能。
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..geometry import (
    poly_get_samples,
    polyline_iou,
    _resample_polyline,
    _compute_xyxy_bbox,
    _normalize_points,
    _xyxy_to_cxcywh_norm
)
from .maptr_assigner import get_maptr_assigner


def align_det_and_gt_by_hungarian(det_polylines, det_scores, det_types, gt_polylines, gt_types, class_names):
    """
    使用匈牙利算法匹配检测和GT的折线
    
    Args:
        det_polylines: 检测折线列表
        det_scores: 检测分数列表
        det_types: 检测类型列表
        gt_polylines: GT折线列表
        gt_types: GT类型列表
        class_names: 类别名称列表
        
    Returns:
        aligned_polylines: 对齐后的折线列表
        aligned_scores: 对齐后的分数列表
    """
    aligned_polylines = [np.zeros((0, 2)) for _ in range(len(gt_polylines))]
    aligned_scores = np.zeros(len(gt_polylines))
    
    for class_name in class_names:
        cls_det_indices = [i for i, t in enumerate(det_types) if t == class_name]
        cls_gt_indices = [i for i, t in enumerate(gt_types) if t == class_name]
        
        if not cls_gt_indices:
            continue
            
        cls_det_polylines = [det_polylines[i] for i in cls_det_indices]
        cls_det_scores = [det_scores[i] for i in cls_det_indices] if det_scores else [1.0]*len(cls_det_indices)
        cls_gt_polylines = [gt_polylines[i] for i in cls_gt_indices]
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(cls_gt_polylines), len(cls_det_polylines)+len(cls_gt_polylines)))
        for i, gt_poly in enumerate(cls_gt_polylines):
            for j, det_poly in enumerate(cls_det_polylines):
                det_samples = poly_get_samples(det_poly, num_samples=100)
                iou_matrix[i, j] = polyline_iou(gt_poly, det_poly, det_samples)
            # 对角线元素设置为小的正值以确保每个GT都有一个匹配
            iou_matrix[i, len(cls_det_polylines)+i] = 0.1
        
        # 解决分配问题
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        # 存储匹配结果
        for gt_idx, det_idx in zip(row_ind, col_ind):
            original_gt_idx = cls_gt_indices[gt_idx]
            if det_idx < len(cls_det_polylines):
                aligned_polylines[original_gt_idx] = cls_det_polylines[det_idx]
                aligned_scores[original_gt_idx] = cls_det_scores[det_idx]
            else:
                aligned_polylines[original_gt_idx] = cls_gt_polylines[gt_idx]
                aligned_scores[original_gt_idx] = 0.0
                
    return aligned_polylines, aligned_scores


def align_det_and_gt_by_maptr_assigner(det_polylines, det_types, gt_polylines, gt_types, class_names, 
                                       pc_range=None, num_sample_points=50, det_scores=None):
    """
    使用 MapTRAssigner 进行预测与GT的匹配，对折线进行重采样与归一化，输出与GT顺序对齐的预测折线与分数。
    
    Args:
        det_polylines: 预测折线列表
        det_types: 预测类别名称列表
        det_scores: 预测得分列表（可选，用于输出对齐后的分数）
        gt_polylines: GT折线列表
        gt_types: GT类别名称列表
        class_names: 参与评估的类别名（确定类别id顺序）
        pc_range: [xmin,ymin,zmin,xmax,ymax,zmax]，用于归一化
        num_sample_points: 每条线重采样的点数
        
    Returns:
        aligned_polylines: 对齐后的折线列表
        aligned_scores: 对齐后的分数列表
    """
    if pc_range is None:
        pc_range = [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0]
    name_to_id = {name: idx for idx, name in enumerate(class_names)}

    num_pred = len(det_polylines)
    num_gt = len(gt_polylines)
    if num_gt == 0:
        return [np.zeros((0, 2), dtype=np.float32)] * 0, np.zeros(0, dtype=np.float32)

    # 构造 GT 数据
    gt_pts_list = []
    gt_bboxes_list = []
    gt_labels_list = []
    for poly, tname in zip(gt_polylines, gt_types):
        resampled = _resample_polyline(poly, num_points=num_sample_points)
        gt_pts_list.append(resampled)
        gt_bboxes_list.append(_compute_xyxy_bbox(resampled))
        gt_labels_list.append(name_to_id.get(tname, -1))

    gt_pts = np.stack(gt_pts_list, axis=0).astype(np.float32)  # (G,P,2)
    # 增加 orders 维度=1 => (G,1,P,2)
    gt_pts = gt_pts[:, None, :, :]
    gt_bboxes = np.stack(gt_bboxes_list, axis=0).astype(np.float32)
    gt_labels = np.array(gt_labels_list, dtype=np.int64)

    # 构造 Pred 数据
    pred_pts_list = []
    pred_bboxes_list = []
    pred_logits = np.zeros((num_pred, len(class_names)), dtype=np.float32)
    det_scores = det_scores if det_scores is not None else [1.0] * num_pred
    for i, (poly, tname) in enumerate(zip(det_polylines, det_types)):
        resampled = _resample_polyline(poly, num_points=num_sample_points)
        pred_pts_list.append(resampled)
        pred_bboxes_list.append(_compute_xyxy_bbox(resampled))
        cls_id = name_to_id.get(tname, -1)
        if 0 <= cls_id < pred_logits.shape[1]:
            try:
                pred_logits[i, cls_id] = float(det_scores[i])
            except Exception:
                pred_logits[i, cls_id] = 1.0

    # 归一化 points 与 bbox_pred(cxcywh)
    pred_pts = np.stack(pred_pts_list, axis=0).astype(np.float32)
    pred_pts_norm = _normalize_points(pred_pts.reshape(-1, 2), pc_range).reshape(pred_pts.shape)
    pred_bbox_cxcywh_norm = np.stack([
        _xyxy_to_cxcywh_norm(xyxy, pc_range) for xyxy in pred_bboxes_list
    ], axis=0).astype(np.float32)

    # torch 张量
    device = torch.device('cpu')
    bbox_pred = torch.from_numpy(pred_bbox_cxcywh_norm).to(device)
    cls_pred = torch.from_numpy(pred_logits).to(device)
    pts_pred = torch.from_numpy(pred_pts_norm).to(device)  # 注意：与训练一致，pred传[0,1]
    gt_bboxes_t = torch.from_numpy(gt_bboxes).to(device)
    gt_labels_t = torch.from_numpy(gt_labels).to(device)
    gt_pts_t = torch.from_numpy(_normalize_points(gt_pts.reshape(-1, 2), pc_range).reshape(gt_pts.shape)).to(device)

    # 使用简化的MapTR分配器
    try:
        assigner = get_maptr_assigner(pc_range=pc_range)
        
        assign_result, order_index = assigner.assign(
            bbox_pred=bbox_pred,
            cls_pred=cls_pred,
            pts_pred=pts_pred,
            gt_bboxes=gt_bboxes_t,
            gt_labels=gt_labels_t,
            gt_pts=gt_pts_t
        )
        
        # 从AssignResult对象中提取所需的值
        assigned_gt_inds = assign_result.gt_inds
        assigned_labels = assign_result.labels

        # 反向对齐：为每个 GT 找匹配的 Pred
        aligned_polylines = [np.zeros((0, 2), dtype=np.float32) for _ in range(num_gt)]
        aligned_scores = np.zeros(num_gt, dtype=np.float32)
        assigned = assigned_gt_inds.cpu().numpy()  # (num_pred,)
        # 由于一对一匹配，遍历预测找到对应的gt
        for pred_idx, gt_ind_plus1 in enumerate(assigned):
            if gt_ind_plus1 <= 0:
                continue
            gt_idx = int(gt_ind_plus1 - 1)
            aligned_polylines[gt_idx] = det_polylines[pred_idx]
            # 若有分数可用，赋值；否则置1
            try:
                aligned_scores[gt_idx] = float(det_scores[pred_idx])
            except Exception:
                aligned_scores[gt_idx] = 1.0

        return aligned_polylines, aligned_scores
        
    except Exception as e:
        print(f"Warning: MapTRAssigner failed ({e}), falling back to Hungarian algorithm")
        return align_det_and_gt_by_hungarian(det_polylines, det_scores, det_types, gt_polylines, gt_types, class_names)
