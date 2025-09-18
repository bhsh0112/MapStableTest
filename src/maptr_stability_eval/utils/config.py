"""
配置和参数解析模块

包含命令行参数解析和配置管理功能。
"""

import argparse


def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='MapTR稳定性评估工具 - 评估pkl或npz格式预测结果的稳定性')
    
    # 数据格式选择
    parser.add_argument('--data-format', type=str, choices=['pkl', 'npz'], default='pkl',
                       help='数据格式：pkl（单个文件）或npz（文件夹） (默认: pkl)')
    
    # 必需参数
    parser.add_argument('--prediction-file', type=str, required=True,
                       help='预测结果文件路径（pkl文件或npz文件夹）')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径，定义输入文件字段格式')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出结果目录 (默认: outputs)')
    
    # 稳定性评估参数
    parser.add_argument('--stability-classes', type=str, nargs='+',
                       default=['divider', 'ped_crossing', 'boundary'],
                       help='评估的类别列表 (默认: divider ped_crossing boundary)')
    parser.add_argument('--stability-interval', type=int, default=2,
                       help='帧间隔，用于配对连续帧 (默认: 2)')
    parser.add_argument('--localization-weight', type=float, default=0.5,
                       help='位置稳定性权重 [0,1]，综合稳定性=在场一致性×(loc*W + shape*(1-W)) (默认: 0.5)')
    
    # 数据处理参数
    parser.add_argument('--detection-threshold', type=float, default=0.3,
                       help='检测阈值，低于此值的预测视为未检测到 (默认: 0.3)')
    parser.add_argument('--pc-range', type=float, nargs=6, 
                       default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                    # default=[-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
                       help='点云范围 [xmin,ymin,zmin,xmax,ymax,zmax] (默认: -25.0 -25.0 -5.0 25.0 25.0 5.0)')
    parser.add_argument('--num-sample-points', type=int, default=50,
                       help='折线重采样点数 (默认: 50)')
    
    # 预测坐标系调整参数（可选）
    parser.add_argument('--pred-rotate-deg', type=float, default=0.0, 
                       help='对预测点先旋转角度（度, 绕Z轴） (默认: 0.0)')
    parser.add_argument('--pred-swap-xy', action='store_true', 
                       help='对预测点先交换 x/y 坐标')
    parser.add_argument('--pred-flip-x', action='store_true', 
                       help='对预测点先翻转 x 轴')
    parser.add_argument('--pred-flip-y', action='store_true', 
                       help='对预测点先翻转 y 轴')
    
    # 其他参数
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出信息')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.localization_weight < 0 or args.localization_weight > 1:
        raise ValueError("localization_weight 必须在 [0, 1] 范围内")
    
    if args.detection_threshold < 0 or args.detection_threshold > 1:
        raise ValueError("detection_threshold 必须在 [0, 1] 范围内")
    
    if args.stability_interval < 1:
        raise ValueError("stability_interval 必须大于 0")
    
    if len(args.pc_range) != 6:
        raise ValueError("pc_range 必须包含 6 个值 [xmin,ymin,zmin,xmax,ymax,zmax]")
    
    return args