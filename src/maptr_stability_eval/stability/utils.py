"""
稳定性评估工具函数

包含结果打印、格式化等辅助功能。
"""

from tabulate import tabulate


def print_stability_index_results(metrics, class_names):
    """
    打印稳定性指标结果
    
    Args:
        metrics: 稳定性指标字典
        class_names: 类别名称列表
        
    Returns:
        metrics_str: 格式化的结果字符串
    """
    metrics_str = '\n----------------------------------\n'
    metrics_str += 'MapTR Stability Index Results\n'
    metrics_str += '----------------------------------\n'

    metrics_data_print = []
    for class_name in [*class_names, 'mean']:
        # 安全获取指标值，如果不存在则使用0.0
        si_key = f'STABILITY_INDEX_{class_name}'
        pc_key = f'PRESENCE_CONSISTENCY_{class_name}'
        lv_key = f'LOCALIZATION_VARIATION_{class_name}'
        sv_key = f'SHAPE_VARIATION_{class_name}'
        
        metrics_data_print.append([
            class_name,
            f"{metrics.get(si_key, 0.0):.4f}",
            f"{metrics.get(pc_key, 0.0):.4f}",
            f"{metrics.get(lv_key, 0.0):.4f}",
            f"{metrics.get(sv_key, 0.0):.4f}"
        ])
    
    metrics_str += tabulate(
        metrics_data_print,
        headers=['class', 'SI', 'presence', 'localization', 'shape'],
        tablefmt='orgtbl'
    )
    return metrics_str
