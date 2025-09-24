#!/usr/bin/env python3
"""
MapTR稳定性评估工具使用示例

展示如何使用新的npz文件加载功能。
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def example_pkl_usage():
    """PKL格式使用示例"""
    print("=" * 60)
    print("PKL格式使用示例")
    print("=" * 60)
    
    cmd = """
python main.py \\
    --data-format pkl \\
    --prediction-file /path/to/your/maptr_results.pkl \\
    --config configs/maptr.py \\
    --output-dir outputs \\
    --stability-classes divider ped_crossing boundary \\
    --stability-interval 2 \\
    --localization-weight 0.5 \\
    --detection-threshold 0.3 \\
    --pc-range -25.0 -25.0 -5.0 25.0 25.0 5.0 \\
    --num-sample-points 50 \\
    --verbose
"""
    
    print("命令示例:")
    print(cmd)
    print("\n说明:")
    print("- 使用 --data-format pkl 指定pkl格式")
    print("- --prediction-file 指向单个pkl文件")
    print("- --config 使用maptr.py配置文件")


def example_npz_usage():
    """NPZ格式使用示例"""
    print("\n" + "=" * 60)
    print("NPZ格式使用示例")
    print("=" * 60)
    
    cmd = """
python main.py \\
    --data-format npz \\
    --prediction-file /path/to/your/npz_folder/ \\
    --config configs/pivotnet.py \\
    --output-dir outputs \\
    --stability-classes divider ped_crossing boundary \\
    --stability-interval 2 \\
    --localization-weight 0.5 \\
    --detection-threshold 0.3 \\
    --pc-range -25.0 -25.0 -5.0 25.0 25.0 5.0 \\
    --num-sample-points 50 \\
    --verbose
"""
    
    print("命令示例:")
    print(cmd)
    print("\n说明:")
    print("- 使用 --data-format npz 指定npz格式")
    print("- --prediction-file 指向包含npz文件的文件夹")
    print("- --config 使用pivotnet.py配置文件")
    print("- 文件夹中每个npz文件对应一个token")


def example_npz_data_format():
    """NPZ数据格式说明"""
    print("\n" + "=" * 60)
    print("NPZ数据格式说明")
    print("=" * 60)
    
    print("NPZ文件结构:")
    print("data/")
    print("├── token1.npz")
    print("├── token2.npz")
    print("├── token3.npz")
    print("└── ...")
    
    print("\n每个npz文件内容:")
    print("```")
    print("import numpy as np")
    print("")
    print("# 创建示例数据")
    print("pts_3d = np.array([")
    print("    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # 第一条折线")
    print("    [[3.0, 3.0], [4.0, 4.0]]               # 第二条折线")
    print("])")
    print("labels_3d = np.array([0, 1])  # 类别标签: 0=divider, 1=ped_crossing, 2=boundary")
    print("scores_3d = np.array([0.8, 0.9])  # 预测分数")
    print("")
    print("# 保存为npz文件")
    print("np.savez('token1.npz',")
    print("         pts_3d=pts_3d,")
    print("         labels_3d=labels_3d,")
    print("         scores_3d=scores_3d)")
    print("```")
    
    print("\n类别映射:")
    print("- 0: divider (车道分隔线)")
    print("- 1: ped_crossing (人行横道)")
    print("- 2: boundary (道路边界)")


def example_configuration():
    """配置文件说明"""
    print("\n" + "=" * 60)
    print("配置文件说明")
    print("=" * 60)
    
    print("1. PKL格式配置文件 (configs/maptr.py):")
    print("   - 适用于MapTR模型输出的pkl文件")
    print("   - 字段映射: pts_3d, labels_3d, scores_3d, sample_idx")
    print("   - 与StableMap中的test_stable.py完全一致")
    
    print("\n2. NPZ格式配置文件 (configs/pivotnet.py):")
    print("   - 适用于PivotNet等模型输出的npz文件")
    print("   - 字段映射: pts_3d, labels_3d, scores_3d, sample_idx")
    print("   - 支持从文件名提取token")
    
    print("\n3. 自定义配置文件:")
    print("   - 可以创建自己的配置文件")
    print("   - 需要定义field_mapping, class_mapping等")
    print("   - 参考现有配置文件的结构")


def example_docker_usage():
    """Docker环境使用示例"""
    print("\n" + "=" * 60)
    print("Docker环境使用示例")
    print("=" * 60)
    
    print("1. 进入Docker容器:")
    print("   docker exec -it MapSerious bash")
    
    print("\n2. 激活虚拟环境:")
    print("   conda activate mapptr")
    
    print("\n3. 进入项目目录:")
    print("   cd /path/to/maptr_stability_eval")
    
    print("\n4. 运行评估:")
    print("   # PKL格式")
    print("   python main.py --data-format pkl --prediction-file results.pkl --config configs/maptr.py")
    print("")
    print("   # NPZ格式")
    print("   python main.py --data-format npz --prediction-file npz_folder/ --config configs/pivotnet.py")
    
    print("\n5. 测试npz加载器:")
    print("   python test_npz_loader.py")


def main():
    """主函数"""
    print("MapTR稳定性评估工具 - 使用示例")
    
    example_pkl_usage()
    example_npz_usage()
    example_npz_data_format()
    example_configuration()
    example_docker_usage()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("✓ 支持PKL格式（单个文件）")
    print("✓ 支持NPZ格式（文件夹）")
    print("✓ 兼容MapTR和PivotNet等模型输出")
    print("✓ 提供完整的配置和测试功能")
    print("✓ 支持Docker环境运行")


if __name__ == '__main__':
    main()

