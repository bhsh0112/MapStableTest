"""
折线处理工具函数

包含折线采样、IoU计算、重采样等折线相关的几何计算功能。
"""

import numpy as np


def poly_get_samples(poly, num_samples=100):
    """
    获取折线的x方向采样点
    
    Args:
        poly: 折线点集，格式为[[x1,y1], [x2,y2], ...]
        num_samples: 采样点数，默认为100
        
    Returns:
        x_samples: x方向的等距采样点数组
    """
    # if len(poly) == 0:
    #     # 如果折线为空，返回默认范围
    #     return np.linspace(-30, 30, num_samples)
    
    x = [p[0] for p in poly]
    # if len(x) == 0:
    #     # 如果x坐标为空，返回默认范围
    #     return np.linspace(-30, 30, num_samples)
    
    min_x, max_x = min(x), max(x)
    
    # 如果min_x == max_x，添加小的偏移避免除零
    if min_x == max_x:
        min_x -= 0.1
        max_x += 0.1
    
    # 在当前帧的范围内生成等距采样点
    x_samples = np.linspace(min_x, max_x, num_samples)
    return x_samples


def polyline_iou(poly1, poly2, x_samples):
    """
    计算两条折线之间的IoU（基于x方向等距离采样）
    
    Args:
        poly1, poly2: 折线点集，格式为[[x1,y1], [x2,y2], ...]
        x_samples: x方向采样点数组
        
    Returns:
        iou: 两条折线的IoU值，范围[0,1]，特殊情况下返回-1
    """
    # 特殊标记，做平均时不算
    if len(poly1) == 0 or len(poly2) == 0:
        return -1
    
    # 对两条折线进行插值
    y1_samples = interpolate_polyline(poly1, x_samples)
    y2_samples = interpolate_polyline(poly2, x_samples)
    
    # 计算绝对差值和
    total_abs_diff = np.sum(np.abs(y1_samples - y2_samples))
    
    # 计算IoU (使用与原始函数相同的归一化因子)
    iou = 1 - total_abs_diff / (len(x_samples) * 15.0)
    
    return max(0.0, min(1.0, iou))


def interpolate_polyline(poly, x_samples):
    """
    在给定的x采样点上对折线进行线性插值
    
    Args:
        poly: 折线点集
        x_samples: 采样点的x坐标
        
    Returns:
        y_samples: 插值得到的y坐标数组
    """
    if len(poly) == 0:
        return np.zeros_like(x_samples)
    
    if len(poly) == 1:
        return np.full_like(x_samples, poly[0][1])
    
    y_samples = np.zeros_like(x_samples)
    segments = list(zip(poly[:-1], poly[1:]))
    current_segment = 0
    
    for i, x in enumerate(x_samples):
        # 找到包含当前x的线段
        while (current_segment < len(segments) and 
               x > segments[current_segment][1][0]):
            current_segment += 1
        
        if current_segment >= len(segments):
            # 超出最后一点，使用最后一个点的y值
            y_samples[i] = poly[-1][1]
        elif x < segments[current_segment][0][0]:
            # 在第一点之前，使用第一个点的y值
            y_samples[i] = poly[0][1]
        else:
            # 在当前线段内进行线性插值
            p0, p1 = segments[current_segment]
            x0, y0 = p0
            x1, y1 = p1
            
            if x1 == x0:  # 垂直线段处理
                y_samples[i] = (y0 + y1) / 2.0
            else:
                t = (x - x0) / (x1 - x0)
                y_samples[i] = y0 + t * (y1 - y0)
    
    return y_samples


def _resample_polyline(points, num_points=50):
    """
    沿弧长对折线均匀重采样到固定点数
    
    Args:
        points: 折线点集 (N,2)
        num_points: 目标点数
        
    Returns:
        resampled_points: 重采样后的点集 (num_points,2)
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts, num_points, axis=0)
    
    segs = pts[1:] - pts[:-1]
    dists = np.sqrt((segs**2).sum(axis=1))
    lengths = np.concatenate([[0.0], np.cumsum(dists)])
    total = lengths[-1]
    
    if total <= 1e-6:
        return np.repeat(pts[:1], num_points, axis=0)
    
    target = np.linspace(0.0, total, num_points)
    res = np.zeros((num_points, 2), dtype=np.float32)
    j = 0
    
    for i, t in enumerate(target):
        while j < len(dists) and lengths[j+1] < t:
            j += 1
        if j >= len(dists):
            res[i] = pts[-1]
        else:
            t0, t1 = lengths[j], lengths[j+1]
            ratio = 0.0 if (t1 - t0) <= 1e-6 else (t - t0) / (t1 - t0)
            res[i] = pts[j] + ratio * (pts[j+1] - pts[j])
    
    return res


def _compute_xyxy_bbox(points):
    """
    计算折线的XYXY边界框
    
    Args:
        points: 折线点集 (N,2)
        
    Returns:
        bbox: 边界框 [x1,y1,x2,y2]
    """
    if points is None or len(points) == 0:
        return np.array([0., 0., 0., 0.], dtype=np.float32)
    
    pts = np.asarray(points, dtype=np.float32)
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _xyxy_to_cxcywh_norm(xyxy, pc_range):
    """
    XYXY -> 归一化 CXCYWH
    
    Args:
        xyxy: 边界框 (4,)
        pc_range: 点云范围 [xmin, ymin, zmin, xmax, ymax, zmax]
        
    Returns:
        cxcywh_norm: 归一化的中心点坐标和宽高 (4,) cx,cy,w,h ∈ [0,1]
    """
    xmin, ymin, xmax, ymax = xyxy
    w = max(0.0, float(xmax - xmin))
    h = max(0.0, float(ymax - ymin))
    cx = float(xmin + xmax) / 2.0
    cy = float(ymin + ymax) / 2.0
    W = float(pc_range[3] - pc_range[0])
    H = float(pc_range[4] - pc_range[1])
    cx_n = (cx - pc_range[0]) / max(W, 1e-6)
    cy_n = (cy - pc_range[1]) / max(H, 1e-6)
    w_n = w / max(W, 1e-6)
    h_n = h / max(H, 1e-6)
    return np.array([cx_n, cy_n, w_n, h_n], dtype=np.float32)


def _normalize_points(points, pc_range):
    """
    按 pc_range 线性归一化到[0,1]
    
    Args:
        points: 点集 (N,2)
        pc_range: 点云范围 [xmin, ymin, zmin, xmax, ymax, zmax]
        
    Returns:
        normalized_points: 归一化后的点集 (N,2)
    """
    pts = np.asarray(points, dtype=np.float32)
    W = float(pc_range[3] - pc_range[0])
    H = float(pc_range[4] - pc_range[1])
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] - pc_range[0]) / max(W, 1e-6)
    out[:, 1] = (pts[:, 1] - pc_range[1]) / max(H, 1e-6)
    return out
