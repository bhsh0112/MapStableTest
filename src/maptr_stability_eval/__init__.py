"""
MapTR Stability Evaluation Package

一个用于评估MapTR模型稳定性的工具包，包含几何计算、稳定性指标计算、
数据解析等功能模块。

主要模块：
- geometry: 几何计算相关功能
- stability: 稳定性评估指标
- data_parser: 数据解析和对齐
- utils: 通用工具函数
"""

__version__ = "1.0.0"
__author__ = "Zhiqi Li"

# 导入主要模块
from . import geometry
from . import stability
from . import data_parser
from . import utils

# 导入常用函数
from .geometry import (
    poly_get_samples,
    polyline_iou,
    interpolate_polyline,
    transform_points_between_frames,
    _apply_pred_local_adjust
)

from .stability import (
    compute_presence_consistency,
    get_localization_variations,
    get_shape_variations,
    eval_maptr_stability_index,
    print_stability_index_results
)

from .data_parser import (
    load_prediction_results,
    parse_prediction_for_stability,
    _safe_dump_outputs
)

from .utils import (
    parse_args
)
