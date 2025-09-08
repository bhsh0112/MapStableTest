"""
稳定性评估模块

包含MapTR模型稳定性评估相关的功能，包括在场一致性、位置稳定性、
形状稳定性等指标的计算。
"""

from .metrics import (
    compute_presence_consistency,
    get_localization_variations,
    get_shape_variations,
    eval_maptr_stability_index
)

from .alignment import (
    align_det_and_gt_by_hungarian,
    align_det_and_gt_by_maptr_assigner
)

from .utils import (
    print_stability_index_results
)

__all__ = [
    'compute_presence_consistency',
    'get_localization_variations', 
    'get_shape_variations',
    'eval_maptr_stability_index',
    'align_det_and_gt_by_hungarian',
    'align_det_and_gt_by_maptr_assigner',
    'print_stability_index_results'
]
