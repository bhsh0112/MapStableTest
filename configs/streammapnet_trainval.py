"""
StreamMapNet 稳定性评估配置文件

适用于 StreamMapNet 模型输出的 pkl 文件格式，与稳定性可视化/评估脚本保持一致。
基于 /data2/file_swap/sh_space/map_test/srcs/StreamMapNet/tools/test_stable.py 的输出约定：
支持包含 `labels`、`vectors`、`scores`、`token/sample_idx` 等字段的结果。
"""

import numpy as np

# StreamMapNet 稳定性评估配置
config = {
    # 字段映射配置 - 基于 StreamMapNet 推理输出结构
    'field_mapping': {
        # 必需字段
        'required_fields': [
            'vectors',       # StreamMapNet 输出中的折线/矢量 (list[np.ndarray] 或等价结构)
            'labels',        # 类别标签 (list[int] / np.ndarray[int])
            'sample_idx'     # 样本索引 (str)，若无则使用 'token'
        ],

        # 字段名映射 - 兼容 StreamMapNet 输出
        'polylines_field': 'vectors',          # 折线字段
        'types_field': 'labels',               # 类别ID字段
        'scores_field': 'scores',              # 预测分数字段（可选）
        'instance_ids_field': 'instance_ids',  # 实例ID（可选，部分流程会从GT侧对齐获得）
        'scene_id_field': 'scene_token',       # 场景ID（可选）
        'timestamp_field': 'timestamp',        # 时间戳（可选）

        # 位姿/标识（可选，通常从 NuScenes 元信息获取）
        'ego_translation_field': 'ego_translation',
        'ego_rotation_field': 'ego_rotation',
        'token_field': 'token'
    },

    # 类别映射 - 与稳定性可视化默认类别一致
    'class_mapping': {
        0: 'divider',        # 车道分隔线
        1: 'ped_crossing',   # 人行横道
        2: 'boundary'        # 道路边界
    },

    # 坐标变换配置 - 对预测进行局部坐标调整（若需要，可在脚本参数中覆盖）
    'coordinate_transform': {
        'rotate_deg': 0.0,        # 旋转角度（度）
        'swap_xy': False,         # 是否交换 x/y 坐标（StreamMapNet 若与 NuScenes 坐标不一致可改为 True）
        'flip_x': False,          # 是否翻转 x 轴
        'flip_y': False           # 是否翻转 y 轴
    },

    # 稳定性评估配置 - 与 StableMap/可视化脚本一致
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],
        'interval': 2,                       # 帧间隔
        'localization_weight': 0.5,          # 位置稳定性权重
        'detection_threshold': 0.3,          # 在场一致性检测阈值
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],  # 点云/BEV范围
        'num_sample_points': 50              # 折线重采样点数
    },

    # NuScenes 数据集配置
    'nuscenes': {
        'version': 'v1.0-trainval',
        'dataroot': 'data/nuscenes',
        'verbose': False
    },

    # 数据验证配置
    'validation': {
        'min_polylines_per_sample': 0,
        'max_polylines_per_sample': 1000,
        'min_polyline_points': 2,
        'max_polyline_points': 1000,
        'valid_types': [0, 1, 2],
        'score_range': [0.0, 1.0]
    },

    # 数据处理配置 - 与稳定性评估实现中的默认值保持一致
    'data_processing': {
        'detection_threshold': 0.01,  # 与 test_stable.py 中的 DETECTION_THRESHOLD 对齐
        'presence_threshold': 0.3,    # 在场一致性阈值
        'x_range': (-30, 30),
        'y_range': (-15, 15)
    }
}

# 数据格式说明（示例）
"""
StreamMapNet 推理输出 pkl 结构：

每个 pkl 文件包含一个列表，列表中每个元素是一个字典，代表一个样本的预测结果。

字段约定：
- vectors: 折线集合，list[np.ndarray]，每个 ndarray 形状约为 (P_i, 2)
- labels: 类别ID，list[int] 或 np.ndarray[int]，取值 {0: divider, 1: ped_crossing, 2: boundary}
- scores: 预测分数，list[float] 或 np.ndarray[float]（可选，缺省按 1.0 处理）
- sample_idx: 样本索引（str），若不存在可使用 token
- token: NuScenes sample token（可选）
- scene_token / timestamp / ego_translation / ego_rotation: 可选，通常从 NuScenes 元数据获取

示例：
[
    {
        'vectors': [np.array([[x1, y1], [x2, y2], ...], dtype=float), ...],
        'labels': np.array([2, 0, 0, ...], dtype=int),
        'scores': np.array([0.71, 0.64, ...], dtype=float),  # 可选
        'sample_idx': '30e55a3ec6184d8cb1944b39ba19d622',
        'token': '30e55a3ec6184d8cb1944b39ba19d622'  # 可选
    },
    ...
]
"""


