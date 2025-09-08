"""
示例配置文件

定义pkl格式预测结果文件的字段映射和稳定性评估参数。
"""

# 配置文件
config = {
    # 字段映射配置 - 定义pkl文件中各字段的名称
    'field_mapping': {
        # 必需字段
        'required_fields': [
            'polylines',      # 折线数据
            'types',          # 类别标签
            'scores',         # 预测分数
            'sample_idx',     # 样本索引
            'timestamp'       # 时间戳
        ],
        
        # 字段名映射（如果pkl文件中的字段名与默认不同，可以在这里映射）
        'polylines_field': 'polylines',           # 折线数据字段名
        'types_field': 'types',                   # 类别标签字段名
        'scores_field': 'scores',                 # 预测分数字段名
        'instance_ids_field': 'instance_ids',     # 实例ID字段名（可选）
        'scene_id_field': 'scene_token',          # 场景ID字段名
        'timestamp_field': 'timestamp',           # 时间戳字段名
        
        # 可选字段
        'ego_translation_field': 'ego_translation',  # 自车位置字段名
        'ego_rotation_field': 'ego_rotation',        # 自车旋转字段名
        'token_field': 'token'                       # token字段名
    },
    
    # 坐标变换配置（可选）
    'coordinate_transform': {
        'rotate_deg': 0.0,        # 旋转角度（度）
        'swap_xy': False,         # 是否交换x/y坐标
        'flip_x': False,          # 是否翻转x轴
        'flip_y': False           # 是否翻转y轴
    },
    
    # 稳定性评估配置
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],  # 评估类别
        'interval': 2,                                        # 帧间隔
        'localization_weight': 0.5,                          # 位置稳定性权重
        'detection_threshold': 0.3,                          # 检测阈值
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],  # 点云范围
        'num_sample_points': 50                              # 折线重采样点数
    },
    
    # 数据验证配置
    'validation': {
        'min_polylines_per_sample': 0,      # 每个样本最少折线数
        'max_polylines_per_sample': 1000,   # 每个样本最多折线数
        'min_polyline_points': 2,           # 每条折线最少点数
        'max_polyline_points': 1000,        # 每条折线最多点数
        'valid_types': ['divider', 'ped_crossing', 'boundary'],  # 有效类别
        'score_range': [0.0, 1.0]          # 分数范围
    }
}

# 数据格式说明
"""
PKL文件格式说明：

每个pkl文件应包含一个列表，列表中的每个元素是一个字典，代表一个样本的预测结果。

字典应包含以下字段：

必需字段：
- polylines: 折线列表，每个折线是一个numpy数组，形状为(N, 2)，表示N个点的x,y坐标
- types: 类别标签列表，长度与polylines相同，每个元素是字符串类型
- scores: 预测分数列表，长度与polylines相同，每个元素是浮点数
- sample_idx: 样本索引，整数类型
- timestamp: 时间戳，用于排序和配对连续帧

可选字段：
- instance_ids: 实例ID列表，用于跨帧匹配同一实例
- scene_token: 场景标识符，用于分组
- ego_translation: 自车位置，numpy数组，形状为(3,)
- ego_rotation: 自车旋转，可以是四元数对象或旋转矩阵
- token: 样本token

示例数据格式：
[
    {
        'polylines': [
            np.array([[0, 0], [1, 1], [2, 0]]),  # 第一条折线
            np.array([[0, 1], [1, 2], [2, 1]])   # 第二条折线
        ],
        'types': ['divider', 'boundary'],
        'scores': [0.8, 0.9],
        'sample_idx': 0,
        'timestamp': 1000,
        'instance_ids': ['divider_1', 'boundary_1'],
        'scene_token': 'scene_001'
    },
    ...
]
"""