#!/usr/bin/env python3
"""
简单使用示例

展示如何使用MapTR稳定性评估工具包。
"""

import numpy as np
import pickle
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root / "src"))

from maptr_stability_eval.stability import eval_maptr_stability_index, print_stability_index_results
from maptr_stability_eval.data_parser import load_prediction_results, parse_prediction_for_stability


def create_sample_data():
    """创建示例数据"""
    print("创建示例数据...")
    
    # 创建示例预测结果
    sample_data = []
    
    for i in range(10):  # 10个样本
        # 创建一些示例折线
        polylines = []
        types = []
        scores = []
        
        # 添加一些随机折线
        for j in range(3):
            # 创建简单的折线
            poly = np.array([
                [i * 2 + j, 0],
                [i * 2 + j + 1, 1],
                [i * 2 + j + 2, 0]
            ])
            polylines.append(poly)
            types.append(['divider', 'ped_crossing', 'boundary'][j])
            scores.append(0.8 + np.random.random() * 0.2)
        
        sample_data.append({
            'polylines': polylines,
            'types': types,
            'scores': scores,
            'sample_idx': i,
            'timestamp': i * 100,  # 时间戳
            'instance_ids': [f'instance_{i}_{j}' for j in range(3)],
            'scene_token': 'scene_001'
        })
    
    return sample_data


def create_sample_config():
    """创建示例配置"""
    config = {
        'field_mapping': {
            'required_fields': ['polylines', 'types', 'scores', 'sample_idx', 'timestamp'],
            'polylines_field': 'polylines',
            'types_field': 'types',
            'scores_field': 'scores',
            'instance_ids_field': 'instance_ids',
            'scene_id_field': 'scene_token',
            'timestamp_field': 'timestamp'
        },
        'stability_eval': {
            'classes': ['divider', 'ped_crossing', 'boundary'],
            'interval': 2,
            'localization_weight': 0.5,
            'detection_threshold': 0.3,
            'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
            'num_sample_points': 50
        }
    }
    return config


def main():
    """主函数"""
    print("MapTR稳定性评估工具包 - 简单示例")
    print("=" * 50)
    
    # 创建示例数据
    sample_data = create_sample_data()
    
    # 保存示例数据到pkl文件
    pkl_file = "sample_data.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(sample_data, f)
    print(f"✓ 示例数据已保存到: {pkl_file}")
    
    # 创建配置
    config = create_sample_config()
    
    # 加载预测结果
    print("\n加载预测结果...")
    prediction_results = load_prediction_results(pkl_file, config)
    
    # 解析数据
    print("\n解析数据...")
    parsed_data = parse_prediction_for_stability(prediction_results, config, interval=2)
    
    # 计算稳定性指标
    print("\n计算稳定性指标...")
    si_dict = eval_maptr_stability_index(
        *parsed_data,
        class_names=['divider', 'ped_crossing', 'boundary'],
        localization_weight=0.5
    )
    
    # 打印结果
    print("\n稳定性评估结果:")
    si_str = print_stability_index_results(si_dict, ['divider', 'ped_crossing', 'boundary'])
    print(si_str)
    
    # 清理临时文件
    if os.path.exists(pkl_file):
        os.remove(pkl_file)
        print(f"\n✓ 临时文件已清理: {pkl_file}")
    
    print("\n示例完成！")


if __name__ == '__main__':
    main()
