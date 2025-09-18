"""
MapTR稳定性评估配置文件

适用于MapTR模型输出的pkl文件格式，与StableMap中的test_stable.py完全一致。
基于 /data2/file_swap/sh_space/map_test/results/MapTRv1/maptr_r18_gkt_24e.pkl 文件格式。
"""

import numpy as np

# MapTR稳定性评估配置
config = {
    # 字段映射配置 - 基于实际pkl文件结构
    'field_mapping': {
        # 必需字段
        'required_fields': [
            'pts_3d',         # MapTR输出中的折线数据 (torch.Tensor)
            'labels_3d',      # MapTR输出中的类别标签 (torch.Tensor)
            'scores_3d',      # MapTR输出中的预测分数 (torch.Tensor)
            'sample_idx'      # 样本索引 (str)
        ],
        
        # 字段名映射 - 与MapTR输出完全一致
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
    
    # 类别映射 - MapTR使用数字标签，与StableMap一致
    'class_mapping': {
        0: 'divider',        # 车道分隔线
        1: 'ped_crossing',   # 人行横道
        2: 'boundary'        # 道路边界
    },
    
    # 坐标变换配置 - 与StableMap默认设置一致
    'coordinate_transform': {
        'rotate_deg': 0.0,        # 旋转角度（度）
        'swap_xy': False,         # 是否交换x/y坐标
        'flip_x': False,          # 是否翻转x轴
        'flip_y': False           # 是否翻转y轴
    },
    
    # 稳定性评估配置 - 与StableMap完全一致
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],  # 评估类别，与StableMap一致
        'interval': 2,                                        # 帧间隔，与StableMap默认值一致
        'localization_weight': 0.5,                          # 位置稳定性权重，与StableMap默认值一致
        'detection_threshold': 0.3,                          # 在场一致性检测阈值，与StableMap一致
        # 'pc_range': [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],  # 点云范围，与StableMap一致
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],  # 点云范围，与StableMap一致
        'num_sample_points': 50                              # 折线重采样点数，与StableMap一致
    },
    
    # NuScenes数据集配置
    'nuscenes': {
        'version': 'v1.0-mini',                          # 数据集版本，pkl文件来自此版本
        'dataroot': 'data/nuscenes',                         # 数据集根目录
        'verbose': False                                     # 是否显示详细加载信息
    },
    
    # 数据验证配置
    'validation': {
        'min_polylines_per_sample': 0,      # 每个样本最少折线数
        'max_polylines_per_sample': 1000,   # 每个样本最多折线数
        'min_polyline_points': 2,           # 每条折线最少点数
        'max_polyline_points': 1000,        # 每条折线最多点数
        'valid_types': [0, 1, 2],           # MapTR使用数字标签
        'score_range': [0.0, 1.0]          # 分数范围
    },
    
    # 数据处理配置 - 与StableMap一致
    'data_processing': {
        'detection_threshold': 0.01,        # 检测阈值，与StableMap中的DETECTION_THRESHOLD一致
        'presence_threshold': 0.3,          # 在场一致性阈值，与StableMap一致
        'x_range': (-30, 30),               # x方向范围，与StableMap一致
        'y_range': (-15, 15)                # y方向范围，与StableMap一致
    }
}

# 数据格式说明
"""
PKL文件格式说明（基于maptr_r18_gkt_24e.pkl）：

每个pkl文件包含一个列表，列表中的每个元素是一个字典，代表一个样本的预测结果。

字典包含以下字段：

必需字段：
- pts_3d: 折线数据，torch.Tensor，形状为(N, P, 2)，N为折线数量，P为每条折线的点数
- labels_3d: 类别标签，torch.Tensor，形状为(N,)，每个元素是0、1或2
- scores_3d: 预测分数，torch.Tensor，形状为(N,)，每个元素是浮点数
- sample_idx: 样本索引，字符串类型

类别映射：
- 0: divider (车道分隔线)
- 1: ped_crossing (人行横道)  
- 2: boundary (道路边界)

示例数据格式：
[
    {
        'pts_3d': torch.Tensor([[x1, y1], [x2, y2], ...], ...),  # 多条折线
        'labels_3d': torch.Tensor([2, 0, 0, 0, 0, 2, ...]),      # 类别标签
        'scores_3d': torch.Tensor([0.4775, 0.4453, ...]),        # 预测分数
        'sample_idx': '30e55a3ec6184d8cb1944b39ba19d622'        # 样本索引
    },
    ...
]
"""
