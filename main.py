#!/usr/bin/env python3
"""
MapTR稳定性评估工具主程序

用于评估MapTR模型在连续帧间的稳定性表现。
直接加载pkl格式的推测结果进行评估。
"""

import os
import sys
import datetime
import argparse
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from maptr_stability_eval.utils import parse_args
from maptr_stability_eval.data_parser import load_prediction_results, _safe_dump_outputs
from maptr_stability_eval.stability import eval_maptr_stability_index, print_stability_index_results


def main():
    """主函数"""
    args = parse_args()

    # 验证参数
    if not args.prediction_file:
        raise ValueError("必须指定预测结果文件 --prediction-file")
    
    if not args.config:
        raise ValueError("必须指定配置文件 --config")
    
    if not os.path.exists(args.prediction_file):
        raise ValueError(f"预测结果文件不存在: {args.prediction_file}")
    
    if not os.path.exists(args.config):
        raise ValueError(f"配置文件不存在: {args.config}")

    print("MapTR稳定性评估工具")
    print("=" * 50)
    print(f"预测结果文件: {args.prediction_file}")
    print(f"配置文件: {args.config}")
    print(f"输出目录: {args.output_dir}")
    print(f"稳定性类别: {args.stability_classes}")
    print(f"帧间隔: {args.stability_interval}")
    print(f"位置权重: {args.localization_weight}")
    print("=" * 50)

    # 加载配置文件
    print("\n加载配置文件...")
    config = load_config(args.config)
    print(f"✓ 配置文件加载成功")

    # 加载预测结果
    print("\n加载预测结果...")
    prediction_results = load_prediction_results(args.prediction_file, config)
    print(f"✓ 预测结果加载成功，共 {len(prediction_results)} 个样本")
    # print(prediction_results[0])

    # 初始化NuScenes（从配置文件或命令行参数）
    nusc = None
    nuscenes_config = config.get('nuscenes', {})
    data_root = args.data_root if hasattr(args, 'data_root') and args.data_root else nuscenes_config.get('dataroot')
    nusc_version = args.nusc_version if hasattr(args, 'nusc_version') and args.nusc_version else nuscenes_config.get('version', 'v1.0-trainval')
    
    if data_root:
        try:
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=nuscenes_config.get('verbose', False))
            print(f"✓ NuScenes数据集初始化成功 (版本: {nusc_version})")
        except Exception as e:
            print(f"Warning: 无法初始化NuScenes数据集: {e}")
            print("将使用默认的ego pose信息")
    else:
        print("Warning: 未提供NuScenes数据路径，将使用默认的ego pose信息")
    
    # 解析数据为稳定性评估格式
    print("\n解析数据...")
    parsed_data = parse_prediction_data(
        prediction_results, config, args.stability_interval, nusc,
        pred_rotate_deg=getattr(args, 'pred_rotate_deg', 0.0),
        pred_swap_xy=getattr(args, 'pred_swap_xy', False),
        pred_flip_x=getattr(args, 'pred_flip_x', False),
        pred_flip_y=getattr(args, 'pred_flip_y', False)
    )
    print(f"✓ 数据解析完成")
    # print(parsed_data[2])

    # 计算稳定性指标
    print("\n计算稳定性指标...")
    si_dict = eval_maptr_stability_index(
        *parsed_data,
        class_names=args.stability_classes,
        localization_weight=float(max(0.0, min(1.0, args.localization_weight)))
    )
    print(f"✓ 稳定性指标计算完成")

    # 打印结果
    print("\n稳定性评估结果:")
    si_str = print_stability_index_results(si_dict, args.stability_classes)
    print(si_str)

    # 保存结果
    if args.output_dir:
        print(f"\n保存结果到 {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = Path(args.prediction_file).stem
        log_file = os.path.join(args.output_dir, f"{base_name}_stability_{timestamp}.log")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"MapTR稳定性评估结果\n")
            f.write(f"预测文件: {args.prediction_file}\n")
            f.write(f"配置文件: {args.config}\n")
            f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"参数设置:\n")
            f.write(f"  - 稳定性类别: {args.stability_classes}\n")
            f.write(f"  - 帧间隔: {args.stability_interval}\n")
            f.write(f"  - 位置权重: {args.localization_weight}\n")
            f.write("\n" + si_str)
        
        print(f"✓ 结果已保存到: {log_file}")

    print("\n评估完成！")


def load_config(config_path):
    """加载配置文件"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if hasattr(config_module, 'config'):
        return config_module.config
    else:
        raise ValueError("配置文件中必须包含 'config' 变量")


def parse_prediction_data(prediction_results, config, interval, nusc=None, 
                         pred_rotate_deg=0.0, pred_swap_xy=False, 
                         pred_flip_x=False, pred_flip_y=False):
    """解析预测数据为稳定性评估格式"""
    from maptr_stability_eval.data_parser import parse_prediction_for_stability
    from maptr_stability_eval.data_parser import parse_maptr_data
    
    # return parse_prediction_for_stability(
    #     prediction_results, config, interval, nusc,
    #     pred_rotate_deg=pred_rotate_deg,
    #     pred_swap_xy=pred_swap_xy,
    #     pred_flip_x=pred_flip_x,
    #     pred_flip_y=pred_flip_y
    # )
    return parse_maptr_data(
        nusc, prediction_results, interval,
        pred_rotate_deg=pred_rotate_deg,
        pred_swap_xy=pred_swap_xy,
        pred_flip_x=pred_flip_x,
        pred_flip_y=pred_flip_y
    )


if __name__ == '__main__':
    main()