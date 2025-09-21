"""
StreamMapNet 稳定性评估配置文件（PKL）

适配 /map_test/results/StreamMapNet/streammapnet_r50_bevformer_30e.pkl 的结构：
- 每个元素为 dict，至少包含: 'pts_3d'(N,P,2 ndarray), 'labels_3d'(N,), 'scores_3d'(N,),
  'sample_idx'(int), 'token'(str)。

该配置会在加载阶段将 sample_idx 统一为 NuScenes 的 sample_token 字符串，避免
"Sample token not found" 的问题。
"""

import numpy as np


def _normalize_streammapnet_records(prediction_results):
    """将每条记录的 sample_idx 规范为字符串 token。

    - 若存在 'token' 且非空，则以 'token' 作为 sample_idx。
    - 否则保留原 sample_idx（假定已为字符串）。
    - 保证 pts_3d/labels_3d/scores_3d 为 numpy/torch 可被下游接受的形状。
    """
    for rec in prediction_results:
        # 将 token 映射为 sample_idx（NuScenes sample_token）
        if 'token' in rec and isinstance(rec['token'], str) and len(rec['token']) > 0:
            rec['sample_idx'] = rec['token']
        # 兜底处理：确保关键字段存在
        if 'pts_3d' not in rec or 'labels_3d' not in rec:
            continue
        # 统一 numpy 类型，保持 (N,P,2)
        pts = rec['pts_3d']
        if isinstance(pts, np.ndarray):
            # 确保为 float32
            rec['pts_3d'] = pts.astype(np.float32, copy=False)
        # 分数兜底
        if 'scores_3d' not in rec or rec['scores_3d'] is None:
            num = len(rec['labels_3d']) if hasattr(rec['labels_3d'], '__len__') else 0
            rec['scores_3d'] = np.ones((num,), dtype=np.float32)


# StreamMapNet 稳定性评估配置
config = {
    # 自定义的预处理钩子：在加载完成后调用以规范 sample_idx 等
    'post_load_hook': _normalize_streammapnet_records,

    # 字段映射配置
    'field_mapping': {
        'required_fields': [
            'pts_3d',        # 折线数据 (numpy.ndarray 或 torch.Tensor)
            'labels_3d',     # 类别标签 (numpy.ndarray 或 torch.Tensor)
            'scores_3d',     # 置信分数
            'sample_idx'     # NuScenes sample_token（字符串）
        ],
        'polylines_field': 'pts_3d',
        'types_field': 'labels_3d',
        'scores_field': 'scores_3d',
        'instance_ids_field': 'instance_ids',
        'scene_id_field': 'scene_token',
        'timestamp_field': 'timestamp',
        'ego_translation_field': 'ego_translation',
        'ego_rotation_field': 'ego_rotation',
        'token_field': 'token'
    },

    # 类别映射（与评估内部一致）
    'class_mapping': {
        0: 'divider',
        1: 'ped_crossing',
        2: 'boundary'
    },

    # 坐标系变换（按需在命令行通过 --pred-swap-xy / --pred-flip-y 控制）
    'coordinate_transform': {
        'rotate_deg': 0.0,
        'swap_xy': False,
        'flip_x': False,
        'flip_y': False
    },

    # 稳定性评估参数
    'stability_eval': {
        'classes': ['divider', 'ped_crossing', 'boundary'],
        'interval': 2,
        'localization_weight': 0.5,
        'pc_range': [-25.0, -25.0, -5.0, 25.0, 25.0, 5.0],
        'num_sample_points': 50
    },

    # NuScenes 数据集
    'nuscenes': {
        'version': 'v1.0-trainval',
        'dataroot': 'data/nuscenes',
        'verbose': False
    },

    # 数据验证
    'validation': {
        'min_polylines_per_sample': 0,
        'max_polylines_per_sample': 1000,
        'min_polyline_points': 2,
        'max_polyline_points': 1000,
        'valid_types': [0, 1, 2],
        'score_range': [0.0, 1.0]
    },

    # 数据处理
    'data_processing': {
        'detection_threshold': 0.01,
        'presence_threshold': 0.3,
        'x_range': (-30, 30),
        'y_range': (-15, 15)
    }
}

