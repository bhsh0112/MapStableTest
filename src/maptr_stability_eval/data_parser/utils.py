"""
数据解析工具函数

包含文件操作、数据保存等辅助功能。
"""

import os
import pickle
import torch


def _ensure_dir_for_file(file_path):
    """
    确保保存结果的父目录存在
    
    Args:
        file_path: 文件路径
    """
    dirpath = os.path.dirname(file_path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def _move_tensors_to_cpu(obj):
    """
    递归地将嵌套结构中的 CUDA Tensor 移动到 CPU，保持原有容器类型
    
    Args:
        obj: 可能包含Tensor的对象
        
    Returns:
        obj_on_cpu: 所有Tensor都移动到CPU的对象
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_cpu(v) for v in obj)
    return obj


def _safe_dump_outputs(outputs, out_path):
    """
    安全地将推理结果保存为 pkl：
    1) 创建输出目录；2) 将所有 Tensor 移动到 CPU；
    3) 先尝试用 mmcv.dump 保存，失败则回退到 pickle.dump
    
    Args:
        outputs: 推理结果
        out_path: 输出路径
    """
    _ensure_dir_for_file(out_path)
    obj_on_cpu = _move_tensors_to_cpu(outputs)
    # 直接使用pickle保存
    with open(out_path, 'wb') as f:
        pickle.dump(obj_on_cpu, f, protocol=pickle.HIGHEST_PROTOCOL)
