#!/usr/bin/env python3
"""
MapTR稳定性评估工具使用示例

展示如何使用该工具包进行稳定性评估。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from maptr_stability_eval.stability import eval_maptr_stability_index, print_stability_index_results
from maptr_stability_eval.data_parser import parse_maptr_data
from maptr_stability_eval.geometry import polyline_iou, poly_get_samples
from nuscenes.nuscenes import NuScenes
import numpy as np


def example_basic_usage():
    """基本使用示例"""
    print("=== MapTR稳定性评估工具基本使用示例 ===")
    
    # 示例数据路径（需要根据实际情况修改）
    config_path = "configs/example_config.py"
    checkpoint_path = "checkpoints/maptr_model.pth"
    data_root = "/path/to/nuscenes"
    nusc_version = "v1.0-mini"
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return
    
    if not os.path.exists(data_root):
        print(f"数据根目录不存在: {data_root}")
        return
    
    # 构建命令行参数
    cmd_args = [
        config_path,
        checkpoint_path,
        "--eval-stability",
        "--data-root", data_root,
        "--nusc-version", nusc_version,
        "--stability-classes", "divider", "ped_crossing", "boundary",
        "--stability-interval", "2",
        "--localization-weight", "0.5",
        "--out", "results/example_results.pkl"
    ]
    
    print("命令行参数:")
    print(" ".join(cmd_args))
    print("\n运行命令:")
    print(f"python main.py {' '.join(cmd_args)}")


def example_geometry_usage():
    """几何计算使用示例"""
    print("\n=== 几何计算使用示例 ===")
    
    # 创建示例折线
    poly1 = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
    poly2 = np.array([[0, 0.1], [1, 1.1], [2, 0.1], [3, 1.1]])
    
    print(f"折线1: {poly1}")
    print(f"折线2: {poly2}")
    
    # 计算采样点
    x_samples = poly_get_samples(poly1, num_samples=10)
    print(f"采样点: {x_samples}")
    
    # 计算IoU
    iou = polyline_iou(poly1, poly2, x_samples)
    print(f"折线IoU: {iou:.4f}")


def example_stability_metrics():
    """稳定性指标计算示例"""
    print("\n=== 稳定性指标计算示例 ===")
    
    # 模拟检测结果
    cur_det_annos = [
        {
            'polylines': [np.array([[0, 0], [1, 1], [2, 0]])],
            'types': ['divider'],
            'scores': [0.8]
        }
    ]
    
    pre_det_annos = [
        {
            'polylines': [np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 0.1]])],
            'types': ['divider'],
            'scores': [0.7]
        }
    ]
    
    # 模拟GT数据
    cur_gt_annos = [
        {
            'polylines': [np.array([[0, 0], [1, 1], [2, 0]])],
            'instance_ids': np.array(['divider_1']),
            'types': np.array(['divider']),
            'ego_translation': np.array([0, 0, 0]),
            'ego_rotation': None  # 需要Quaternion对象
        }
    ]
    
    pre_gt_annos = [
        {
            'polylines': [np.array([[0, 0], [1, 1], [2, 0]])],
            'instance_ids': np.array(['divider_1']),
            'types': np.array(['divider']),
            'ego_translation': np.array([0, 0, 0]),
            'ego_rotation': None  # 需要Quaternion对象
        }
    ]
    
    print("模拟数据已创建")
    print("注意: 实际使用时需要完整的GT数据，包括ego_rotation等")


def example_configuration():
    """配置示例"""
    print("\n=== 配置示例 ===")
    
    config_example = {
        'stability_eval': {
            'classes': ['divider', 'ped_crossing', 'boundary'],
            'interval': 2,
            'localization_weight': 0.5,
            'detection_threshold': 0.3,
            'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
            'num_sample_points': 50
        }
    }
    
    print("稳定性评估配置示例:")
    for key, value in config_example['stability_eval'].items():
        print(f"  {key}: {value}")


def example_output_format():
    """输出格式示例"""
    print("\n=== 输出格式示例 ===")
    
    # 模拟稳定性指标结果
    metrics = {
        'STABILITY_INDEX_divider': 0.8234,
        'STABILITY_INDEX_ped_crossing': 0.7891,
        'STABILITY_INDEX_boundary': 0.8567,
        'STABILITY_INDEX_mean': 0.8231,
        'PRESENCE_CONSISTENCY_divider': 0.9123,
        'PRESENCE_CONSISTENCY_ped_crossing': 0.8765,
        'PRESENCE_CONSISTENCY_boundary': 0.9234,
        'PRESENCE_CONSISTENCY_mean': 0.9041,
        'LOCALIZATION_VARIATION_divider': 0.8456,
        'LOCALIZATION_VARIATION_ped_crossing': 0.8123,
        'LOCALIZATION_VARIATION_boundary': 0.8678,
        'LOCALIZATION_VARIATION_mean': 0.8419,
        'SHAPE_VARIATION_divider': 0.7891,
        'SHAPE_VARIATION_ped_crossing': 0.7456,
        'SHAPE_VARIATION_boundary': 0.8234,
        'SHAPE_VARIATION_mean': 0.7860
    }
    
    class_names = ['divider', 'ped_crossing', 'boundary']
    
    # 打印结果
    result_str = print_stability_index_results(metrics, class_names)
    print(result_str)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MapTR稳定性评估工具使用示例')
    parser.add_argument('--example', type=str, 
                       choices=['basic', 'geometry', 'metrics', 'config', 'output', 'all'],
                       default='all',
                       help='要运行的示例类型')
    
    args = parser.parse_args()
    
    if args.example == 'basic' or args.example == 'all':
        example_basic_usage()
    
    if args.example == 'geometry' or args.example == 'all':
        example_geometry_usage()
    
    if args.example == 'metrics' or args.example == 'all':
        example_stability_metrics()
    
    if args.example == 'config' or args.example == 'all':
        example_configuration()
    
    if args.example == 'output' or args.example == 'all':
        example_output_format()
    
    print("\n=== 示例完成 ===")
    print("更多信息请参考README.md文档")


if __name__ == '__main__':
    main()
