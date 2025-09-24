#!/usr/bin/env python3
"""
NPZ加载功能演示脚本

展示如何使用maptr_stability_eval项目加载PivotNet等模型输出的npz文件。
每个npz文件对应一个token，支持从文件夹中批量加载。
"""

import os
import sys
import numpy as np
import tempfile
import shutil
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from maptr_stability_eval.data_parser import (
    load_npz_prediction_results,
    get_npz_file_info
)
from maptr_stability_eval.utils import parse_args


def create_pivotnet_style_npz_files(output_dir, num_files=5):
    """
    创建PivotNet风格的npz文件
    
    Args:
        output_dir: 输出目录
        num_files: 创建的文件数量
    """
    print(f"创建PivotNet风格的npz文件到目录: {output_dir}")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        token = f"pivotnet_token_{i:03d}"
        npz_file = os.path.join(output_dir, f"{token}.npz")
        
        # 创建模拟的PivotNet输出数据
        # 随机生成2-6条折线
        num_polylines = np.random.randint(2, 7)
        polylines = []
        labels = []
        scores = []
        
        for j in range(num_polylines):
            # 每条折线3-10个点
            num_points = np.random.randint(3, 11)
            # 生成在合理范围内的点（模拟车道线、人行横道等）
            if j % 3 == 0:  # 车道分隔线
                points = np.random.uniform(-15, 15, (num_points, 2))
                points[:, 1] = np.random.uniform(-2, 2, num_points)  # y方向变化较小
            elif j % 3 == 1:  # 人行横道
                points = np.random.uniform(-10, 10, (num_points, 2))
                points[:, 0] = np.random.uniform(-1, 1, num_points)  # x方向变化较小
            else:  # 道路边界
                points = np.random.uniform(-20, 20, (num_points, 2))
            
            polylines.append(points)
            
            # 随机类别标签 (0, 1, 2)
            label = np.random.randint(0, 3)
            labels.append(label)
            
            # 随机分数 (0.1, 1.0)
            score = np.random.uniform(0.1, 1.0)
            scores.append(score)
        
        # 转换为numpy数组（PivotNet输出格式）
        pts_3d = np.array(polylines, dtype=object)
        labels_3d = np.array(labels, dtype=np.int64)
        scores_3d = np.array(scores, dtype=np.float32)
        
        # 保存npz文件
        np.savez(npz_file,
                 pts_3d=pts_3d,
                 labels_3d=labels_3d,
                 scores_3d=scores_3d)
        
        print(f"  ✓ 创建文件: {npz_file}")
        print(f"    折线数: {num_polylines}, 类别: {labels}, 分数范围: [{min(scores):.3f}, {max(scores):.3f}]")
    
    print(f"✓ 成功创建 {num_files} 个PivotNet风格的npz文件")


def demo_npz_loading():
    """演示npz加载功能"""
    print("=" * 80)
    print("NPZ加载功能演示")
    print("=" * 80)
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        npz_dir = os.path.join(temp_dir, "pivotnet_outputs")
        
        # 1. 创建PivotNet风格的npz文件
        print("\n1. 创建PivotNet风格的npz文件")
        create_pivotnet_style_npz_files(npz_dir, num_files=5)
        
        # 2. 获取文件信息
        print("\n2. 获取npz文件信息")
        info = get_npz_file_info(npz_dir)
        if 'error' in info:
            print(f"❌ 获取文件信息失败: {info['error']}")
            return False
        else:
            print(f"✓ 文件信息获取成功:")
            print(f"  总文件数: {info['total_files']}")
            print(f"  示例文件: {info['sample_file']}")
            print(f"  示例token: {info['sample_token']}")
            print(f"  数据键: {info['sample_keys']}")
            print(f"  pts_3d形状: {info['pts_3d_shape']}")
            print(f"  labels_3d形状: {info['labels_3d_shape']}")
            print(f"  scores_3d形状: {info['scores_3d_shape']}")
        
        # 3. 加载配置
        print("\n3. 加载PivotNet配置")
        import importlib.util
        config_path = "configs/pivotnet.py"
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config
        
        print(f"✓ 配置文件加载成功")
        print(f"  字段映射: {config['field_mapping']['polylines_field']} -> pts_3d")
        print(f"  类别映射: {config['class_mapping']}")
        
        # 4. 加载npz预测结果
        print("\n4. 加载npz预测结果")
        try:
            prediction_results = load_npz_prediction_results(npz_dir, config)
            print(f"✓ 成功加载 {len(prediction_results)} 个预测结果")
            
            # 显示第一个结果的详细信息
            if prediction_results:
                sample = prediction_results[0]
                print(f"\n  示例结果详情:")
                print(f"    token: {sample['sample_idx']}")
                print(f"    pts_3d形状: {sample['pts_3d'].shape}")
                print(f"    labels_3d: {sample['labels_3d'].tolist()}")
                print(f"    scores_3d: {sample['scores_3d'].tolist()}")
                print(f"    数据类型: {type(sample['pts_3d'])}")
                
                # 显示类别统计
                labels = sample['labels_3d'].tolist()
                class_counts = {0: 0, 1: 0, 2: 0}
                for label in labels:
                    class_counts[label] += 1
                print(f"    类别统计: divider={class_counts[0]}, ped_crossing={class_counts[1]}, boundary={class_counts[2]}")
            
        except Exception as e:
            print(f"❌ 加载npz预测结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. 演示命令行使用
        print("\n5. 命令行使用示例")
        print("=" * 50)
        print("使用以下命令运行稳定性评估:")
        print(f"python main.py \\")
        print(f"    --data-format npz \\")
        print(f"    --prediction-file {npz_dir} \\")
        print(f"    --config configs/pivotnet.py \\")
        print(f"    --output-dir outputs \\")
        print(f"    --stability-interval 2 \\")
        print(f"    --localization-weight 0.5")
        print("=" * 50)
        
        # 6. 数据格式说明
        print("\n6. NPZ文件格式说明")
        print("=" * 50)
        print("PivotNet输出的npz文件格式:")
        print("- 每个npz文件对应一个token（从文件名提取）")
        print("- pts_3d: 折线数据，形状为(N, P, 2)，N为折线数量，P为每条折线的点数")
        print("- labels_3d: 类别标签，形状为(N,)，0=divider, 1=ped_crossing, 2=boundary")
        print("- scores_3d: 预测分数，形状为(N,)，范围[0,1]")
        print("- 支持不同长度的折线（使用object类型存储）")
        print("=" * 50)
    
    print("\n" + "=" * 80)
    print("✓ NPZ加载功能演示完成！")
    print("✓ 项目已成功支持PivotNet等模型的npz输出格式")
    print("=" * 80)
    return True


def demo_command_line_usage():
    """演示命令行使用"""
    print("\n" + "=" * 80)
    print("命令行使用演示")
    print("=" * 80)
    
    print("1. 基本用法:")
    print("   python main.py --data-format npz --prediction-file /path/to/npz/folder --config configs/pivotnet.py")
    
    print("\n2. 完整参数:")
    print("   python main.py \\")
    print("       --data-format npz \\")
    print("       --prediction-file /path/to/npz/folder \\")
    print("       --config configs/pivotnet.py \\")
    print("       --output-dir outputs \\")
    print("       --stability-classes divider ped_crossing boundary \\")
    print("       --stability-interval 2 \\")
    print("       --localization-weight 0.5 \\")
    print("       --detection-threshold 0.3")
    
    print("\n3. 参数说明:")
    print("   --data-format: 数据格式，npz表示从文件夹加载多个npz文件")
    print("   --prediction-file: npz文件所在目录路径")
    print("   --config: 配置文件，定义字段映射和类别信息")
    print("   --output-dir: 输出结果目录")
    print("   --stability-classes: 评估的类别列表")
    print("   --stability-interval: 帧间隔，用于配对连续帧")
    print("   --localization-weight: 位置稳定性权重[0,1]")
    print("   --detection-threshold: 检测阈值，低于此值的预测视为未检测到")
    
    print("\n4. 配置文件说明:")
    print("   configs/pivotnet.py: 适用于PivotNet等模型的npz输出格式")
    print("   - 字段映射: pts_3d, labels_3d, scores_3d")
    print("   - 类别映射: 0=divider, 1=ped_crossing, 2=boundary")
    print("   - 支持不同长度的折线数据")


if __name__ == '__main__':
    success = demo_npz_loading()
    if success:
        demo_command_line_usage()
        print("\n🎉 演示完成！maptr_stability_eval项目现在完全支持PivotNet的npz输出格式。")
    else:
        print("\n❌ 演示失败！请检查代码问题。")
        sys.exit(1)

