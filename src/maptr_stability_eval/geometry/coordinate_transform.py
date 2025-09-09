"""
坐标变换模块

包含坐标系转换、预测结果调整等坐标变换相关的功能函数。
"""

import numpy as np


def transform_points_between_frames(points, src_translation, src_rotation, 
                                   dst_translation, dst_rotation):
    """
    将点从源坐标系转换到目标坐标系
    
    Args:
        points: 要转换的点集 (N,2)
        src_translation: 源坐标系平移向量
        src_rotation: 源坐标系旋转四元数
        dst_translation: 目标坐标系平移向量  
        dst_rotation: 目标坐标系旋转四元数
        
    Returns:
        transformed_points: 转换后的点集 (N,2)
    """
    # 确保输入是正确的形状
    src_translation = np.array(src_translation).flatten()
    dst_translation = np.array(dst_translation).flatten()
    
    # 转换为齐次坐标 (4维: x, y, 0, 1)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # 现在points是二维数组，每行是(x, y)
    # 添加第三维z=0，然后添加齐次坐标1
    num_points = points.shape[0]
    points_3d = np.column_stack([points, np.zeros(num_points)])
    points_homo = np.column_stack([points_3d, np.ones(num_points)])
    
    # 源坐标系到世界坐标系
    src_to_world = np.eye(4)
    src_to_world[:3, :3] = src_rotation.rotation_matrix
    src_to_world[:3, 3] = src_translation[:3]  # 只取前三个元素
    
    # 世界坐标系到目标坐标系
    world_to_dst = np.eye(4)
    world_to_dst[:3, :3] = dst_rotation.inverse.rotation_matrix
    world_to_dst[:3, 3] = -dst_rotation.inverse.rotate(dst_translation)
    
    # 组合变换
    transform = world_to_dst @ src_to_world
    
    # 应用变换
    transformed = (transform @ points_homo.T).T
    # 返回二维坐标（忽略z坐标）
    return transformed[:, :2]  # 返回二维坐标


def _apply_pred_local_adjust(points, rotate_deg=0.0, swap_xy=False, flip_x=False, flip_y=False):
    """
    对预测的局部坐标进行统一调整（在跨帧/坐标变换前进行）
    
    Args:
        points: 点集 (N,2)
        rotate_deg: 逆时针旋转角度（度）
        swap_xy: 是否交换 x/y
        flip_x: 是否翻转 x
        flip_y: 是否翻转 y
        
    Returns:
        adjusted_points: 调整后的点集 (N,2)
    """
    if points is None or len(points) == 0:
        return points
    
    pts = np.asarray(points, dtype=np.float64)
    
    if swap_xy:
        pts = pts[:, [1, 0]]
    if flip_x:
        pts[:, 0] = -pts[:, 0]
    if flip_y:
        pts[:, 1] = -pts[:, 1]
    if abs(rotate_deg) > 1e-6:
        theta = np.deg2rad(rotate_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts = (R @ pts.T).T
    
    return pts
