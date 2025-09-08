"""
简化的MapTRAssigner实现

从原始MapTRAssigner中提取核心功能，用于稳定性评估中的实例匹配。
最小化依赖，只保留必要的匹配逻辑。
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class AssignResult:
    """分配结果类，模拟MMDetection的AssignResult"""
    
    def __init__(self, num_gts, assigned_gt_inds, max_overlaps=None, labels=None):
        self.num_gts = num_gts
        self.num_preds = assigned_gt_inds.size(0)
        self.gt_inds = assigned_gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels


def bbox_xyxy_to_cxcywh(bboxes):
    """将边界框从xyxy格式转换为cxcywh格式"""
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def bbox_cxcywh_to_xyxy(bboxes):
    """将边界框从cxcywh格式转换为xyxy格式"""
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


class AssignResult:
    """
    存储预测框与真实框的分配结果
    
    Attributes:
        num_gts (int): 真实框的数量
        gt_inds (Tensor): 每个预测框对应的真实框索引，0表示未分配，-1表示忽略
        max_overlaps (Tensor): 每个预测框与其分配的真实框的最大重叠度
        labels (Tensor): 每个预测框对应的真实框的标签
    """
    
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
    
    @property
    def num_preds(self):
        """返回预测框的数量"""
        return len(self.gt_inds)
    
    def add_gt_(self, gt_labels):
        """
        将真实框添加到分配结果中
        
        Args:
            gt_labels (Tensor): 真实框的标签
        """
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])


def bbox_xyxy_to_cxcywh(bboxes):
    """
    将边界框从 (x_min, y_min, x_max, y_max) 格式转换为 (center_x, center_y, width, height) 格式
    
    Args:
        bboxes (Tensor): 形状为 (N, 4) 的张量，每行表示一个边界框，格式为 (x_min, y_min, x_max, y_max)
        
    Returns:
        Tensor: 形状为 (N, 4) 的张量，每行表示一个边界框，格式为 (center_x, center_y, width, height)
    """
    x_min, y_min, x_max, y_max = bboxes.unbind(dim=-1)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return torch.stack((center_x, center_y, width, height), dim=-1)


def bbox_cxcywh_to_xyxy(bboxes):
    """
    将边界框从 (center_x, center_y, width, height) 格式转换为 (x_min, y_min, x_max, y_max) 格式
    
    Args:
        bboxes (Tensor): 形状为 (N, 4) 的张量，每行表示一个边界框，格式为 (center_x, center_y, width, height)
        
    Returns:
        Tensor: 形状为 (N, 4) 的张量，每行表示一个边界框，格式为 (x_min, y_min, x_max, y_max)
    """
    center_x, center_y, width, height = bboxes.unbind(dim=-1)
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)


def normalize_2d_bbox(bboxes, pc_range):
    """归一化2D边界框"""
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    
    # 转换为cxcywh格式
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    
    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    """归一化2D点 - 与MapTRv1实现保持一致"""
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    """反归一化2D边界框"""
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    
    return bboxes


def denormalize_2d_pts(pts, pc_range):
    """反归一化2D点 - 与MapTRv1实现保持一致"""
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return new_pts


class ClassificationCost:
    """分类成本计算"""
    
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def __call__(self, cls_pred, gt_labels):
        """
        计算分类成本
        
        Args:
            cls_pred: 预测分类logits [num_query, num_class]
            gt_labels: GT标签 [num_gt]
            
        Returns:
            cost: 分类成本 [num_query, num_gt]
        """
        # 简化版本：使用负对数似然
        num_query, num_class = cls_pred.shape
        num_gt = gt_labels.shape[0]
        
        # 过滤掉无效标签（-1）
        valid_mask = gt_labels >= 0
        if not valid_mask.any():
            # 如果没有有效标签，返回零成本
            return torch.zeros(num_query, num_gt, device=cls_pred.device)
        
        # 只处理有效标签
        valid_gt_labels = gt_labels[valid_mask]
        valid_indices = torch.where(valid_mask)[0]
        
        # 计算成本矩阵
        cost = torch.zeros(num_query, num_gt, device=cls_pred.device)
        
        # 对每个有效GT标签计算成本
        for i, gt_label in enumerate(valid_gt_labels):
            if 0 <= gt_label < num_class:
                # 使用负对数似然作为成本
                pred_probs = F.softmax(cls_pred, dim=1)
                cost[:, valid_indices[i]] = -torch.log(pred_probs[:, gt_label] + 1e-8)
        
        return cost * self.weight


class BBoxL1Cost:
    """边界框L1成本计算"""
    
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def __call__(self, bbox_pred, gt_bboxes):
        """
        计算边界框L1成本
        
        Args:
            bbox_pred: 预测边界框 [num_query, 4] (cx, cy, w, h)
            gt_bboxes: GT边界框 [num_gt, 4] (cx, cy, w, h)
            
        Returns:
            cost: L1成本 [num_query, num_gt]
        """
        # 计算L1距离
        cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return cost * self.weight


class IoUCost:
    """IoU成本计算"""
    
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def __call__(self, bbox_pred, gt_bboxes):
        """
        计算IoU成本
        
        Args:
            bbox_pred: 预测边界框 [num_query, 4] (x1, y1, x2, y2)
            gt_bboxes: GT边界框 [num_gt, 4] (x1, y1, x2, y2)
            
        Returns:
            cost: IoU成本 [num_query, num_gt]
        """
        # 计算IoU
        iou = self.bbox_overlaps(bbox_pred, gt_bboxes)
        # IoU成本 = 1 - IoU
        cost = 1 - iou
        return cost * self.weight
    
    def bbox_overlaps(self, bboxes1, bboxes2):
        """计算边界框重叠度"""
        # 计算交集
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [num_query, num_gt, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [num_query, num_gt, 2]
        
        wh = (rb - lt).clamp(min=0)  # [num_query, num_gt, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [num_query, num_gt]
        
        # 计算面积
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        # 计算IoU
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)
        
        return iou


class ChamferDistance:
    """Chamfer距离计算"""
    
    def __init__(self, loss_src_weight=1.0, loss_dst_weight=1.0):
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight
    
    def __call__(self, pts_pred, gt_pts):
        """
        计算Chamfer距离
        
        Args:
            pts_pred: 预测点 [num_query, num_pts, 2]
            gt_pts: GT点 [num_gt, num_orders, num_pts, 2]
            
        Returns:
            cost: Chamfer距离成本 [num_query, num_gt, num_orders]
        """
        num_query, num_pts_pred, _ = pts_pred.shape
        num_gt, num_orders, num_pts_gt, _ = gt_pts.shape
        
        # 初始化成本矩阵
        cost = torch.zeros(num_query, num_gt, num_orders, device=pts_pred.device)
        
        # 对每个查询和GT组合计算Chamfer距离
        for q in range(num_query):
            for g in range(num_gt):
                for o in range(num_orders):
                    pred_pts = pts_pred[q]  # [num_pts_pred, 2]
                    gt_pts_single = gt_pts[g, o]  # [num_pts_gt, 2]
                    
                    # 计算所有点对之间的距离
                    dist = torch.cdist(pred_pts.unsqueeze(0), gt_pts_single.unsqueeze(0))  # [1, num_pts_pred, num_pts_gt]
                    dist = dist.squeeze(0)  # [num_pts_pred, num_pts_gt]
                    
                    # 计算Chamfer距离
                    dist1 = dist.min(dim=1)[0].mean()  # 从预测点到GT的最小距离
                    dist2 = dist.min(dim=0)[0].mean()  # 从GT到预测点的最小距离
                    
                    cost[q, g, o] = self.loss_src_weight * dist1 + self.loss_dst_weight * dist2
        
        return cost


class OrderedPtsL1Cost:
    """
    有序点L1成本计算
    
    Args:
        weight (int | float, optional): 成本权重
    """
    
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def __call__(self, bbox_pred, gt_bboxes):
        """
        计算有序点L1成本
        
        Args:
            bbox_pred (Tensor): 预测点坐标，形状 [num_query, num_pts, 2]
            gt_bboxes (Tensor): GT点坐标，形状 [num_gt, num_ordered, num_pts, 2]
            
        Returns:
            torch.Tensor: 成本值 [num_query, num_gt * num_ordered]
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        
        # 将预测点展平
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        
        # 将GT点展平并重新组织
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
        
        # 计算L1距离
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        
        return bbox_cost * self.weight


class SimpleMapTRAssigner:
    """
    简化的MapTR分配器，与MapTRv1的MapTRAssigner完全兼容
    
    Args:
        cls_cost (dict, optional): 分类成本配置
        reg_cost (dict, optional): 回归成本配置  
        iou_cost (dict, optional): IoU成本配置
        pts_cost (dict, optional): 点成本配置
        pc_range (list, optional): 点云范围
    """
    
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pts_cost=dict(type='ChamferDistance', loss_src_weight=1.0, loss_dst_weight=1.0),
                 pc_range=None):
        self.pc_range = pc_range or [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0]
        
        # 根据配置初始化成本计算器
        self.cls_cost = self._build_cost(cls_cost)
        self.reg_cost = self._build_cost(reg_cost)
        self.iou_cost = self._build_cost(iou_cost)
        self.pts_cost = self._build_cost(pts_cost)
    
    def _build_cost(self, cost_config):
        """构建成本计算器"""
        cost_type = cost_config.get('type', 'ClassificationCost')
        weight = cost_config.get('weight', 1.0)
        
        if cost_type == 'ClassificationCost':
            return ClassificationCost(weight=weight)
        elif cost_type == 'BBoxL1Cost':
            return BBoxL1Cost(weight=weight)
        elif cost_type == 'IoUCost':
            return IoUCost(weight=weight)
        elif cost_type == 'ChamferDistance':
            loss_src_weight = cost_config.get('loss_src_weight', 1.0)
            loss_dst_weight = cost_config.get('loss_dst_weight', 1.0)
            return ChamferDistance(loss_src_weight=loss_src_weight, loss_dst_weight=loss_dst_weight)
        elif cost_type == 'OrderedPtsL1Cost':
            return OrderedPtsL1Cost(weight=weight)
        else:
            raise ValueError(f"Unknown cost type: {cost_type}")
    
    def assign(self,
               bbox_pred,
               cls_pred,
               pts_pred,
               gt_bboxes, 
               gt_labels,
               gt_pts,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
        执行分配，与MapTRv1的MapTRAssigner接口完全一致
        
        Args:
            bbox_pred (Tensor): 预测边界框，归一化坐标 (cx, cy, w, h)，范围[0,1]，形状 [num_query, 4]
            cls_pred (Tensor): 预测分类logits，形状 [num_query, num_class]
            pts_pred (Tensor): 预测点，形状 [num_query, num_pts, 2]
            gt_bboxes (Tensor): GT边界框，非归一化坐标 (x1, y1, x2, y2)，形状 [num_gt, 4]
            gt_labels (Tensor): GT标签，形状 (num_gt,)
            gt_pts (Tensor): GT点，形状 [num_gt, num_orders, num_pts, 2]
            gt_bboxes_ignore (Tensor, optional): 忽略的GT边界框，默认为None
            eps (int | float, optional): 数值稳定性参数，默认为1e-7
            
        Returns:
            AssignResult: 分配结果对象
            order_index: 点的顺序索引 [num_query, num_gt]
        """
        # 添加断言检查，与MapTRv1保持一致
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        assert bbox_pred.shape[-1] == 4, \
            'Only support bbox pred shape is 4 dims'
        
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None
        
        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        
        normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :4], normalized_gt_bboxes[:, :4])

        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        num_pts_per_predline = pts_pred.size(1)
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(pts_pred.permute(0,2,1),size=(num_pts_per_gtline),
                                            mode='linear', align_corners=True)
            pts_pred_interpolated = pts_pred_interpolated.permute(0,2,1).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_pts)
        pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)
        
        bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost + pts_cost
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels), order_index


# 全局分配器实例
_maptr_assigner = None


def get_maptr_assigner(pc_range=None, **kwargs):
    """获取MapTR分配器实例，支持与MapTRv1相同的参数"""
    global _maptr_assigner
    if _maptr_assigner is None:
        _maptr_assigner = SimpleMapTRAssigner(pc_range=pc_range, **kwargs)
    return _maptr_assigner


# 为了完全兼容，添加一个别名
MapTRAssigner = SimpleMapTRAssigner
