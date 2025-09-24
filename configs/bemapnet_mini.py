"""
/**
 * BeMapNet 稳定性评估配置文件
 *
 * @model BeMapNet
 * @input_format npz（每个 token 一个文件）
 * @compat PivotNet/MapTR 评估脚本兼容的字段映射
 *
 * 说明：
 * - 适用于 BeMapNet 推理导出的 .npz 结果文件（通常名为 <token>.npz）。
 * - 文件内包含：`dt_mask`（语义/实例掩码）与 `dt_res`（结果字典）。
 * - `dt_res` 至少包含：`map`（折线数组列表）、`confidence_level`（分数列表）、`pred_label`（标签列表）。
 * - 可选透传：`token`、`sample_idx`、`ego_translation`、`ego_rotation`、`extrinsic`、`intrinsic`、`ida_mats`。
 *
 * 标签约定（BeMapNet 默认）
 * - `pred_label` 通常为 [0(占位), 1..C]，其中 1 对应类别 0，以此类推。
 * - 为与 MapTR 统一，可在解析时将 label 减 1，或按下方 class_mapping 解释使用。
 */
"""

import numpy as np

# BeMapNet 稳定性评估配置
config = {
    # 强制所有样本按 BEV 像素坐标处理（像素→米制恒启用）
    'assume_bev_pixels': True,
    # 字段映射配置 - 基于 BeMapNet 实际输出结构
    'field_mapping': {
        # 必需字段（标准化后用于评估）
        'required_fields': [
            'pts_3d',         # 折线数据 (从 dt_res.map 提取)
            'labels_3d',      # 类别标签 (从 dt_res.pred_label 提取，必要时做偏移/映射)
            'scores_3d',      # 预测分数 (从 dt_res.confidence_level 提取)
            'sample_idx'      # 样本索引/标识 (通常从文件名/透传字段获取)
        ],

        # 原始结果字段
        'dt_mask_field': 'dt_mask',             # 分割/实例掩码（形状 ~ (num_classes, H, W)）
        'dt_res_field': 'dt_res',               # 结果字典
        'map_field': 'map',                     # 折线数据 (在 dt_res 中)
        'confidence_field': 'confidence_level', # 置信度分数 (在 dt_res 中)
        'pred_label_field': 'pred_label',       # 预测标签 (在 dt_res 中)

        # 标准字段映射（转换命名以适配统一评估流程）
        'polylines_field': 'pts_3d',
        'types_field': 'labels_3d',
        'scores_field': 'scores_3d',

        # 可选字段（若存在将被透传/用于可视化或坐标变换）
        'ego_translation_field': 'ego_translation',  # (3,) or (1,3)
        'ego_rotation_field': 'ego_rotation',        # (4,) quaternion
        'token_field': 'token',
        'sample_idx_field': 'sample_idx',
        'extrinsic_field': 'extrinsic',
        'intrinsic_field': 'intrinsic',
        'ida_mats_field': 'ida_mats',

        # 标签偏移/修正（供解析器使用，可按需实现）
        'label_offset': 1,                  # BeMapNet 通常使用 1..C，对齐到 0..C-1 时可减 1
        'map_label_3_to_2': True            # 若存在 3 表示 boundary 的别名，映射为 2
    },

    # 类别映射 - 与 MapTR/常用定义保持一致
    'class_mapping': {
        0: 'divider',        # 车道分隔线
        1: 'ped_crossing',   # 人行横道
        2: 'boundary'        # 道路边界
    },

    # 坐标变换配置（如需对折线进行简单坐标修正）
    'coordinate_transform': {
        'rotate_deg': 0.0,
        'swap_xy': False,
        'flip_x': False,
        'flip_y': False
    },

    'bev': {
        'size': (200, 400),           # BEV分辨率 (W,H)
        'radius_x_m': 30.0,           # 半径x（常见：50m）
        'radius_y_m': 15.0,           # 半径y（常见：25m）
        'assume_bev_pixels': True,    # 强制视为BEV像素（绕过启发式）
        'pixel_detect_ratio': 0.0     # 0 表示启发式恒通过（双保险）
    },

    # 启用npz加载器的自动朝向优化（将更多点纳入评估窗口）
    'auto_adjust_orientation': False,

    # 稳定性评估配置
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],  # 评估类别
        'interval': 2,                                        # 帧间隔
        'localization_weight': 0.5,                           # 位置稳定性权重
        'detection_threshold': 0.3,                           # 在场一致性检测阈值
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],    # 点云范围（用于采样/规范化）
        'num_sample_points': 50                               # 折线重采样点数
    },

    # NuScenes 数据集配置（如需基于元数据加载/校验）
    'nuscenes': {
        'version': 'v1.0-mini',
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

    # 数据处理配置
    'data_processing': {
        'detection_threshold': 0.01,
        'presence_threshold': 0.3,
        # 与 BEV 半径对应：x∈[-radius_x_m, radius_x_m], y∈[-radius_y_m, radius_y_m]
        'x_range': (-15, 15),
        'y_range': (-30, 30)
    },

    # npz 文件特定配置
    'npz_config': {
        'file_extension': '.npz',          # 文件扩展名
        'token_from_filename': True,       # 是否从文件名提取 token
        'allow_pickle': True,              # 允许 object 类型（dt_res 通常为 object/dict）
        'default_scores': True,            # 若缺少分数，是否使用默认值 1.0
        'auto_reshape': True               # 自动调整数据形状（如将 (1,3)->(3,)）
    }
}

# 数据格式说明
"""
/**
 * BeMapNet NPZ 文件格式说明：
 *
 * 每个 npz 文件包含一个样本的预测结果，文件名即为 token。
 *
 * 必备键：
 * - dt_mask: numpy.ndarray，形状约为 (C, H, W)，uint8/uint16
 * - dt_res:  dict(object)，包含至少以下字段：
 *   - map:               list[np.ndarray]，每个元素形状 (N, 2)
 *   - confidence_level:  list[float]
 *   - pred_label:        list[int]，通常为 [0, 1..C]
 *
 * 可选透传：
 *   - token:             str
 *   - sample_idx:        str/int
 *   - ego_translation:   np.ndarray/list，形状 (3,) 或 (1,3)
 *   - ego_rotation:      np.ndarray/list，形状 (4,) 或 (1,4)（四元数）
 *   - extrinsic:         np.ndarray，形状 (B, 4, 4) 或单帧 (4,4)
 *   - intrinsic:         np.ndarray，形状 (B, 3, 3) 或单帧 (3,3)
 *   - ida_mats:          np.ndarray，形状 (B, 3, 3) 或单帧 (3,3)
 *
 * 类别映射建议：
 * - 0: divider (车道分隔线)
 * - 1: ped_crossing (人行横道)
 * - 2: boundary (道路边界)
 * - 若 `pred_label` 出现 3 表示 boundary 的别名，可按配置映射为 2。
 */
"""


