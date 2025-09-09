"""
几何计算模块测试

测试折线处理、坐标变换等几何计算功能。
"""

import pytest
import numpy as np
from maptr_stability_eval.geometry import (
    poly_get_samples,
    polyline_iou,
    interpolate_polyline,
    _resample_polyline,
    _compute_xyxy_bbox,
    _normalize_points,
    _xyxy_to_cxcywh_norm,
    transform_points_between_frames,
    _apply_pred_local_adjust
)


class TestPolylineUtils:
    """折线工具函数测试"""
    
    def test_poly_get_samples(self):
        """测试折线采样点生成"""
        poly = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
        samples = poly_get_samples(poly, num_samples=10)
        
        assert len(samples) == 10
        assert samples[0] == 0.0  # 最小值
        assert samples[-1] == 3.0  # 最大值
        assert np.all(np.diff(samples) > 0)  # 递增
    
    def test_interpolate_polyline(self):
        """测试折线插值"""
        poly = np.array([[0, 0], [1, 1], [2, 0]])
        x_samples = np.array([0, 0.5, 1, 1.5, 2])
        y_samples = interpolate_polyline(poly, x_samples)
        
        expected = np.array([0, 0.5, 1, 0.5, 0])
        np.testing.assert_array_almost_equal(y_samples, expected)
    
    def test_polyline_iou(self):
        """测试折线IoU计算"""
        poly1 = np.array([[0, 0], [1, 1], [2, 0]])
        poly2 = np.array([[0, 0.1], [1, 1.1], [2, 0.1]])
        x_samples = poly_get_samples(poly1, num_samples=10)
        
        iou = polyline_iou(poly1, poly2, x_samples)
        assert 0 <= iou <= 1
    
    def test_polyline_iou_empty(self):
        """测试空折线IoU计算"""
        poly1 = np.array([])
        poly2 = np.array([[0, 0], [1, 1]])
        x_samples = np.array([0, 1])
        
        iou = polyline_iou(poly1, poly2, x_samples)
        assert iou == -1
    
    def test_resample_polyline(self):
        """测试折线重采样"""
        poly = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
        resampled = _resample_polyline(poly, num_points=10)
        
        assert resampled.shape == (10, 2)
        assert np.allclose(resampled[0], poly[0])  # 起点
        assert np.allclose(resampled[-1], poly[-1])  # 终点
    
    def test_compute_xyxy_bbox(self):
        """测试边界框计算"""
        points = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
        bbox = _compute_xyxy_bbox(points)
        
        expected = np.array([0, 0, 3, 3])
        np.testing.assert_array_equal(bbox, expected)
    
    def test_compute_xyxy_bbox_empty(self):
        """测试空点集边界框计算"""
        points = np.array([])
        bbox = _compute_xyxy_bbox(points)
        
        expected = np.array([0., 0., 0., 0.])
        np.testing.assert_array_equal(bbox, expected)
    
    def test_normalize_points(self):
        """测试点归一化"""
        points = np.array([[0, 0], [1, 1], [2, 2]])
        pc_range = [0, 0, -5, 2, 2, 5]
        normalized = _normalize_points(points, pc_range)
        
        expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_xyxy_to_cxcywh_norm(self):
        """测试边界框格式转换"""
        xyxy = np.array([0, 0, 2, 2])
        pc_range = [0, 0, -5, 4, 4, 5]
        cxcywh = _xyxy_to_cxcywh_norm(xyxy, pc_range)
        
        expected = np.array([0.25, 0.25, 0.5, 0.5])  # cx, cy, w, h
        np.testing.assert_array_almost_equal(cxcywh, expected)


class TestCoordinateTransform:
    """坐标变换测试"""
    
    def test_apply_pred_local_adjust(self):
        """测试预测结果局部调整"""
        points = np.array([[1, 2], [3, 4]])
        
        # 测试交换xy
        adjusted = _apply_pred_local_adjust(points, swap_xy=True)
        expected = np.array([[2, 1], [4, 3]])
        np.testing.assert_array_equal(adjusted, expected)
        
        # 测试翻转x
        adjusted = _apply_pred_local_adjust(points, flip_x=True)
        expected = np.array([[-1, 2], [-3, 4]])
        np.testing.assert_array_equal(adjusted, expected)
        
        # 测试翻转y
        adjusted = _apply_pred_local_adjust(points, flip_y=True)
        expected = np.array([[1, -2], [3, -4]])
        np.testing.assert_array_equal(adjusted, expected)
    
    def test_apply_pred_local_adjust_rotation(self):
        """测试预测结果旋转"""
        points = np.array([[1, 0], [0, 1]])
        
        # 90度旋转
        adjusted = _apply_pred_local_adjust(points, rotate_deg=90)
        expected = np.array([[0, 1], [-1, 0]])
        np.testing.assert_array_almost_equal(adjusted, expected, decimal=6)
    
    def test_apply_pred_local_adjust_empty(self):
        """测试空点集调整"""
        points = np.array([])
        adjusted = _apply_pred_local_adjust(points)
        assert len(adjusted) == 0
        
        points = None
        adjusted = _apply_pred_local_adjust(points)
        assert adjusted is None


if __name__ == '__main__':
    pytest.main([__file__])
