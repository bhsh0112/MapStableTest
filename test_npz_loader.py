#!/usr/bin/env python3
"""
测试NPZ加载器功能

用于验证npz_loader.py是否能正确加载PivotNet等模型输出的npz文件。
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

from maptr_stability_eval.data_parser.npz_loader import (
    load_npz_prediction_results,
    convert_npz_to_prediction_format,
    validate_npz_prediction_format,
    get_npz_file_info
)


def create_test_npz_files(test_dir, num_files=5):
    """
    创建测试用的npz文件，模拟PivotNet的输出格式
    
    Args:
        test_dir: 测试目录路径
        num_files: 创建的npz文件数量
    """
    print(f"创建测试npz文件到目录: {test_dir}")
    
    # 确保目录存在
    os.makedirs(test_dir, exist_ok=True)
    
    for i in range(num_files):
        token = f"test_token_{i:03d}"
        npz_file = os.path.join(test_dir, f"{token}.npz")
        
        # 创建模拟数据
        # 随机生成2-5条折线
        num_polylines = np.random.randint(2, 6)
        polylines = []
        labels = []
        scores = []
        
        for j in range(num_polylines):
            # 每条折线3-8个点
            num_points = np.random.randint(3, 9)
            # 生成在合理范围内的点
            points = np.random.uniform(-20, 20, (num_points, 2))
            polylines.append(points)
            
            # 随机类别标签 (0, 1, 2)
            label = np.random.randint(0, 3)
            labels.append(label)
            
            # 随机分数 (0.1, 1.0)
            score = np.random.uniform(0.1, 1.0)
            scores.append(score)
        
        # 转换为numpy数组
        pts_3d = np.array(polylines, dtype=object)
        labels_3d = np.array(labels, dtype=np.int64)
        scores_3d = np.array(scores, dtype=np.float32)
        
        # 保存npz文件
        np.savez(npz_file,
                 pts_3d=pts_3d,
                 labels_3d=labels_3d,
                 scores_3d=scores_3d)
        
        print(f"  ✓ 创建文件: {npz_file}")
        print(f"    折线数: {num_polylines}, 类别: {labels}, 分数: {scores[:3]}...")
    
    print(f"✓ 成功创建 {num_files} 个测试npz文件")


def test_npz_loader():
    """测试npz加载器功能"""
    print("=" * 60)
    print("测试NPZ加载器功能")
    print("=" * 60)
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        test_npz_dir = os.path.join(temp_dir, "test_npz")
        
        # 1. 创建测试npz文件
        print("\n1. 创建测试npz文件")
        create_test_npz_files(test_npz_dir, num_files=3)
        
        # 2. 测试获取文件信息
        print("\n2. 测试获取npz文件信息")
        try:
            info = get_npz_file_info(test_npz_dir)
            if 'error' in info:
                print(f"❌ 获取文件信息失败: {info['error']}")
                return False
            else:
                print(f"✓ 文件信息获取成功:")
                print(f"  总文件数: {info['total_files']}")
                print(f"  示例文件: {info['sample_file']}")
                print(f"  示例token: {info['sample_token']}")
                print(f"  数据键: {info['sample_keys']}")
        except Exception as e:
            print(f"❌ 获取文件信息异常: {e}")
            return False
        
        # 3. 测试加载npz预测结果
        print("\n3. 测试加载npz预测结果")
        try:
            # 使用PivotNet配置
            config = {
                'field_mapping': {
                    'polylines_field': 'pts_3d',
                    'types_field': 'labels_3d',
                    'scores_field': 'scores_3d',
                    'required_fields': ['pts_3d', 'labels_3d', 'scores_3d', 'sample_idx']
                }
            }
            
            prediction_results = load_npz_prediction_results(test_npz_dir, config)
            print(f"✓ 成功加载 {len(prediction_results)} 个预测结果")
            
            # 检查第一个结果
            if prediction_results:
                sample = prediction_results[0]
                print(f"  示例结果:")
                print(f"    token: {sample['sample_idx']}")
                print(f"    pts_3d形状: {sample['pts_3d'].shape}")
                print(f"    labels_3d形状: {sample['labels_3d'].shape}")
                print(f"    scores_3d形状: {sample['scores_3d'].shape}")
                print(f"    pts_3d类型: {type(sample['pts_3d'])}")
                print(f"    labels_3d类型: {type(sample['labels_3d'])}")
                print(f"    scores_3d类型: {type(sample['scores_3d'])}")
            
        except Exception as e:
            print(f"❌ 加载npz预测结果失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 测试数据格式验证
        print("\n4. 测试数据格式验证")
        try:
            validate_npz_prediction_format(prediction_results, config)
            print("✓ 数据格式验证通过")
        except Exception as e:
            print(f"❌ 数据格式验证失败: {e}")
            return False
        
        # 5. 测试单个npz文件转换
        print("\n5. 测试单个npz文件转换")
        try:
            # 加载第一个npz文件
            npz_files = [f for f in os.listdir(test_npz_dir) if f.endswith('.npz')]
            if npz_files:
                npz_file = os.path.join(test_npz_dir, npz_files[0])
                npz_data = np.load(npz_file, allow_pickle=True)
                token = Path(npz_file).stem
                
                pred_dict = convert_npz_to_prediction_format(npz_data, token, config)
                if pred_dict:
                    print(f"✓ 单个文件转换成功:")
                    print(f"  token: {pred_dict['sample_idx']}")
                    print(f"  pts_3d形状: {pred_dict['pts_3d'].shape}")
                    print(f"  labels_3d: {pred_dict['labels_3d']}")
                    print(f"  scores_3d: {pred_dict['scores_3d']}")
                else:
                    print("❌ 单个文件转换失败")
                    return False
        except Exception as e:
            print(f"❌ 单个文件转换异常: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！NPZ加载器功能正常")
    print("=" * 60)
    return True


def test_main_integration():
    """测试与main.py的集成"""
    print("\n" + "=" * 60)
    print("测试与main.py的集成")
    print("=" * 60)
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        test_npz_dir = os.path.join(temp_dir, "test_npz")
        
        # 创建测试npz文件
        create_test_npz_files(test_npz_dir, num_files=2)
        
        # 测试命令行参数
        print("\n测试命令行调用:")
        print(f"python main.py \\")
        print(f"    --data-format npz \\")
        print(f"    --prediction-file {test_npz_dir} \\")
        print(f"    --config configs/pivotnet.py \\")
        print(f"    --output-dir {temp_dir}/outputs")
        
        # 这里可以实际调用main.py进行测试
        # 但为了避免依赖问题，我们只显示命令


if __name__ == '__main__':
    success = test_npz_loader()
    if success:
        test_main_integration()
        print("\n🎉 测试完成！NPZ加载器功能正常，可以支持PivotNet等模型的输出。")
    else:
        print("\n❌ 测试失败！需要修复NPZ加载器的问题。")
        sys.exit(1)