"""
PivotNet稳定性评估配置文件

适用于PivotNet等模型输出的npz文件格式，每个npz文件对应一个token。
支持从文件夹中加载多个npz文件进行评估。

PivotNet实际输出格式：
- dt_mask: (3, 400, 200) 分割掩码
- dt_res: 包含map, confidence_level, pred_label, token, ego_translation, ego_rotation
"""

import numpy as np

# PivotNet稳定性评估配置
config = {
    # 字段映射配置 - 基于PivotNet实际输出格式
    'field_mapping': {
        # 必需字段
        'required_fields': [
            'pts_3d',         # 折线数据 (从dt_res.map提取)
            'labels_3d',      # 类别标签 (从dt_res.pred_label提取)
            'scores_3d',      # 预测分数 (从dt_res.confidence_level提取)
            'sample_idx'      # 样本索引 (从文件名提取)
        ],
        
        # PivotNet特有字段
        'dt_mask_field': 'dt_mask',           # 分割掩码
        'dt_res_field': 'dt_res',             # 结果字典
        'map_field': 'map',                   # 折线数据 (在dt_res中)
        'confidence_field': 'confidence_level', # 置信度分数 (在dt_res中)
        'pred_label_field': 'pred_label',     # 预测标签 (在dt_res中)
        
        # 标准字段映射 (用于兼容性)
        'polylines_field': 'pts_3d',           # 转换后的折线数据
        'types_field': 'labels_3d',            # 转换后的类别标签
        'scores_field': 'scores_3d',           # 转换后的预测分数
        
        # 可选字段
        'ego_translation_field': 'ego_translation',  # 在dt_res中
        'ego_rotation_field': 'ego_rotation',        # 在dt_res中
        'token_field': 'token'                       # 在dt_res中
    },
    
    # 类别映射 - 与MapTR保持一致
    'class_mapping': {
        0: 'None',
        1: 'divider',        # 车道分隔线
        2: 'ped_crossing',   # 人行横道
        3: 'boundary'        # 道路边界
    },
    
    # 坐标变换配置
    'coordinate_transform': {
        'rotate_deg': 0.0,        # 旋转角度（度）
        'swap_xy': False,         # 是否交换x/y坐标
        'flip_x': False,          # 是否翻转x轴
        'flip_y': False           # 是否翻转y轴
    },

    # BEV 像素->米制转换半径
    'bev': {
        'radius_x_m': 15.0,
        'radius_y_m': 30.0
    },
    
    # 稳定性评估配置
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],  # 评估类别
        'interval': 2,                                        # 帧间隔
        'localization_weight': 0.5,                          # 位置稳定性权重
        'detection_threshold': 0.3,                          # 在场一致性检测阈值
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],  # 点云范围
        'num_sample_points': 50                              # 折线重采样点数
    },
    
    # NuScenes数据集配置
    'nuscenes': {
        'version': 'v1.0-mini',                          # 数据集版本
        'dataroot': 'data/nuscenes',                         # 数据集根目录
        'verbose': False                                     # 是否显示详细加载信息
    },
    
    # 数据验证配置
    'validation': {
        'min_polylines_per_sample': 0,      # 每个样本最少折线数
        'max_polylines_per_sample': 1000,   # 每个样本最多折线数
        'min_polyline_points': 2,           # 每条折线最少点数
        'max_polyline_points': 1000,        # 每条折线最多点数
        'valid_types': [0, 1, 2],           # 有效类别标签
        'score_range': [0.0, 1.0]          # 分数范围
    },
    
    # 数据处理配置
    'data_processing': {
        'detection_threshold': 0.01,        # 检测阈值
        'presence_threshold': 0.3,          # 在场一致性阈值
        'x_range': (-30, 30),               # x方向范围
        'y_range': (-15, 15)                # y方向范围
    },
    
    # npz文件特定配置
    'npz_config': {
        'file_extension': '.npz',           # 文件扩展名
        'token_from_filename': True,        # 是否从文件名提取token
        'allow_pickle': True,               # 是否允许pickle格式
        'default_scores': True,             # 如果缺少分数，是否使用默认值1.0
        'auto_reshape': True                # 是否自动调整数据形状
    }
}

# 数据格式说明
"""
PivotNet NPZ文件格式说明：

每个npz文件包含一个样本的预测结果，文件名即为token。

PivotNet实际输出格式：
- dt_mask: 分割掩码，numpy.ndarray，形状为(3, 400, 200)，dtype=uint8
- dt_res: 结果字典，numpy.ndarray，dtype=object，包含以下字段：
  - map: 折线数据列表，每个元素是一个numpy.ndarray，形状为(N, 2)
  - confidence_level: 置信度分数列表，每个元素是浮点数
  - pred_label: 预测标签列表，每个元素是整数 (0, 1, 2, 3)
  - token: 样本token字符串
  - ego_translation: ego位置，torch.Tensor，形状为(1, 3)
  - ego_rotation: ego旋转，torch.Tensor，形状为(1, 4)

类别映射：
- 0: divider (车道分隔线)
- 1: ped_crossing (人行横道)  
- 2: boundary (道路边界)
- 3: boundary (道路边界，映射为2)

文件结构示例：
data/
├── 000681a060c04755a1537cf83b53ba57.npz
├── 000868a72138448191b4092f75ed7776.npz
├── 0017c2623c914571a1ff2a37f034ffd7.npz
└── ...

每个npz文件内容示例：
{
    'dt_mask': np.array(shape=(3, 400, 200), dtype=uint8),  # 分割掩码
    'dt_res': {
        'map': [None, array([[x1, y1], [x2, y2], ...]), ...],  # 折线数据
        'confidence_level': [-1, 0.9999, 0.9998, ...],        # 置信度分数
        'pred_label': [0, 1, 1, 2, 3, ...],                    # 预测标签
        'token': '000681a060c04755a1537cf83b53ba57',           # 样本token
        'ego_translation': tensor([[x, y, z]]),                # ego位置
        'ego_rotation': tensor([[qx, qy, qz, qw]])            # ego旋转
    }
}
"""
