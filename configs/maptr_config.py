"""
MapTR标准配置文件

适用于标准MapTR模型输出的pkl文件格式。
"""

import numpy as np

# MapTR标准配置
config = {
    # 字段映射配置
    'field_mapping': {
        'required_fields': [
            'pts_3d',         # MapTR输出中的折线数据
            'labels_3d',      # MapTR输出中的类别标签
            'scores_3d',      # MapTR输出中的预测分数
            'sample_idx'      # 样本索引
        ],
        
        # 字段名映射
        'polylines_field': 'pts_3d',           # MapTR使用pts_3d存储折线
        'types_field': 'labels_3d',            # MapTR使用labels_3d存储类别
        'scores_field': 'scores_3d',           # MapTR使用scores_3d存储分数
        'instance_ids_field': 'instance_ids',  # 实例ID（需要额外提供）
        'scene_id_field': 'scene_token',       # 场景ID
        'timestamp_field': 'timestamp',        # 时间戳
        
        # 可选字段
        'ego_translation_field': 'ego_translation',
        'ego_rotation_field': 'ego_rotation',
        'token_field': 'token'
    },
    
    # 类别映射（MapTR使用数字标签）
    'class_mapping': {
        0: 'divider',
        1: 'ped_crossing', 
        2: 'boundary'
    },
    
    # 坐标变换配置
    'coordinate_transform': {
        'rotate_deg': 0.0,
        'swap_xy': False,
        'flip_x': False,
        'flip_y': False
    },
    
    # 稳定性评估配置
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],
        'interval': 2,
        'localization_weight': 0.5,
        'detection_threshold': 0.3,
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
        'num_sample_points': 50
    },
    
    # 数据验证配置
    'validation': {
        'min_polylines_per_sample': 0,
        'max_polylines_per_sample': 1000,
        'min_polyline_points': 2,
        'max_polyline_points': 1000,
        'valid_types': [0, 1, 2],  # MapTR使用数字标签
        'score_range': [0.0, 1.0]
    }
}
