"""
几何计算模块

包含折线处理、坐标变换、IoU计算等几何相关的功能函数。
"""

from .polyline_utils import (
    poly_get_samples,
    polyline_iou,
    interpolate_polyline,
    _resample_polyline,
    _compute_xyxy_bbox,
    _normalize_points,
    _xyxy_to_cxcywh_norm
)

from .coordinate_transform import (
    transform_points_between_frames,
    _apply_pred_local_adjust
)

__all__ = [
    'poly_get_samples',
    'polyline_iou', 
    'interpolate_polyline',
    '_resample_polyline',
    '_compute_xyxy_bbox',
    '_normalize_points',
    '_xyxy_to_cxcywh_norm',
    'transform_points_between_frames',
    '_apply_pred_local_adjust'
]
