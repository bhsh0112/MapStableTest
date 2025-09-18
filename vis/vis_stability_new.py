#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/**
 * @file vis_stability.py
 * @brief 连续帧语义矢量地图稳定性可视化脚本。
 *
 * 功能：
 * - 读取推理输出（pkl）与 NuScenes 元数据；
 * - 将前若干帧的预测矢量变换到当前帧自车坐标系；
 * - 以颜色与透明度衰减叠加绘制，突出稳定性；
 * - 计算并在图中标注简单的跨帧一致性指标（基于折线IoU的近邻一致性）。
 * - 当启用 GIF 导出时，默认改为在世界坐标系动态计算的紧凑范围内，逐帧仅绘制当前帧结果（不叠加历史）。
 * - 支持 GIF 模式逐帧"累积"显示此前所有帧结果，逐步形成完整地图（默认开启，可通过开关关闭）。
 * - 支持对预测的局部坐标系进行调整（旋转/交换XY/翻转），用于对齐 NuScenes EGO 坐标系。
 * - 分别生成六视图GIF和地图GIF，避免拼接布局问题。
 *
 * 使用示例：
 * python vis_stability.py --pred pred.pkl --data-root /data/nuscenes --gif
 *
 * 输出文件：
 * - scene_map.gif (地图可视化)
 * - scene_six_view.gif (六视图)
 * - map_frames/ (地图帧文件夹)
 * - six_view_frames/ (六视图帧文件夹)
 *
 * 依赖：mmcv, nuscenes, numpy, matplotlib, imageio(optional), shapely(仅在其他脚本中用到，这里不强制)。
 */
"""

import os
import argparse
import datetime
try:
    import mmcv
except Exception:
    mmcv = None
import numpy as np
from typing import Dict, List, Tuple
import pickle

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')  # 后端设为非交互
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.transforms as transforms


# ================== 可视化配置 ==================
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']
CLASS_TO_COLOR = {
    'divider': (0.20, 0.60, 1.00),       # 蓝青
    'ped_crossing': (1.00, 0.55, 0.10),  # 橙
    'boundary': (0.20, 0.80, 0.20),      # 绿
}


def _ensure_dir(d: str) -> None:
    """
    /**
     * @description 确保目录存在
     * @param {str} d 目录路径
     */
    """
    os.makedirs(d, exist_ok=True)


def draw_ego_vehicle(ax: plt.Axes, 
                     ego_translation: np.ndarray,
                     ego_rotation: Quaternion,
                     alpha: float = 1.0,
                     car_size: float = 2.0) -> None:
    """
    /**
     * @description 在指定位置绘制自车图标
     * @param {Axes} ax Matplotlib Axes
     * @param {np.ndarray} ego_translation 自车位置
     * @param {Quaternion} ego_rotation 自车旋转
     * @param {float} alpha 透明度
     * @param {float} car_size 车辆图标大小
     */
    """
    try:
        # 尝试加载自车图标
        car_img = Image.open('figs/car.png')
    except FileNotFoundError:
        print("自车图标不存在，绘制简单的矩形作为车辆")
        car_img = None
    
    if car_img is not None:
        # 使用图标文件
        x, y = ego_translation[0], ego_translation[1]
        rotation_degrees = np.degrees(np.arctan2(ego_rotation.rotation_matrix[1, 0], 
                                                ego_rotation.rotation_matrix[0, 0]))
        
        translation = transforms.Affine2D().translate(x, y)
        rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
        rotation_translation = rotation + translation
        
        ax.imshow(car_img, extent=[-car_size, car_size, -car_size, car_size], 
                 transform=rotation_translation + ax.transData, alpha=alpha)
    else:
        # 绘制简单的矩形车辆
        x, y = ego_translation[0], ego_translation[1]
        rotation_degrees = np.degrees(np.arctan2(ego_rotation.rotation_matrix[1, 0], 
                                                ego_rotation.rotation_matrix[0, 0]))
        
        # 创建车辆矩形
        car_width, car_length = car_size, car_size * 1.5
        car_rect = plt.Rectangle((x - car_width/2, y - car_length/2), car_width, car_length, 
                               angle=rotation_degrees, alpha=alpha, 
                               color='orange', edgecolor='black', linewidth=1)
        ax.add_patch(car_rect)
        
        # 添加车辆方向指示
        direction_length = car_length * 0.6
        dx = direction_length * np.cos(np.radians(rotation_degrees))
        dy = direction_length * np.sin(np.radians(rotation_degrees))
        ax.arrow(x, y, dx, dy, head_width=car_width*0.3, head_length=car_width*0.2, 
                fc='red', ec='red', alpha=alpha)


def _load_outputs(pred_path: str):
    """
    /**
     * @description 读取预测输出pkl（由测试脚本保存）
     * @param {str} pred_path 预测结果pkl路径
     * @returns {Any} outputs 列表，元素包含 sample_idx, labels_3d, pts_3d, scores_3d(可选)
     */
    """
    if mmcv is not None:
        return mmcv.load(pred_path)
    # 兼容无 mmcv 环境
    with open(pred_path, 'rb') as f:
        return pickle.load(f)


def _convert_gt_to_outputs_format(gt_data: dict):
    """
    /**
     * @description 将真值数据转换为与预测输出相同的格式
     * @param {dict} gt_data 真值数据
     * @returns {List[dict]} outputs 格式化的输出列表
     */
    """
    outputs = []
    
    # 遍历真值数据中的每个样本
    for sample_info in gt_data.get('infos', []):
        sample_token = sample_info['token']
        
        # 获取该样本的标注数据
        annotation = sample_info.get('annotation', {})
        
        # 收集所有类别的折线数据
        all_polys = []
        all_labels = []
        
        # 类别映射
        class_mapping = {
            'divider': 0,
            'ped_crossing': 1, 
            'boundary': 2
        }
        
        for class_name, class_id in class_mapping.items():
            if class_name in annotation:
                class_polys = annotation[class_name]
                for poly in class_polys:
                    if len(poly) > 0:
                        all_polys.append(poly)
                        all_labels.append(class_id)
        
        if len(all_polys) == 0:
            continue
            
        # 转换为numpy数组
        pts_3d = np.array(all_polys, dtype=np.float32)
        labels_3d = np.array(all_labels, dtype=np.int64)
        
        # 为真值数据设置默认置信度为1.0
        scores_3d = np.ones(len(pts_3d), dtype=np.float32)
        
        output = {
            'sample_idx': sample_token,
            'pts_3d': pts_3d,
            'labels_3d': labels_3d,
            'scores_3d': scores_3d
        }
        outputs.append(output)
    
    return outputs


def _collect_scene_sequences(nusc: NuScenes, outputs: List[dict], score_thr: float = 0.0,
                             class_filter: List[str] = None) -> Dict[str, List[dict]]:
    """
    /**
     * @description 将预测按 scene_token 分组并按时间排序，补充每帧的自车位姿信息。
     * @param {NuScenes} nusc NuScenes 句柄
     * @param {List[dict]} outputs 推理输出列表
     * @param {float} score_thr 置信度阈值
     * @param {string[]} class_filter 保留类别（None 表示不过滤）
     * @returns {Object} scene_token -> 有序帧列表，每帧含 {token, timestamp, ego_translation, ego_rotation, polys, types, scores}
     */
    """
    class_filter = class_filter or CLASS_NAMES

    scene_to_frames: Dict[str, List[dict]] = {}
    for pred in outputs:
        sample_token = pred['sample_idx']
        try:
            sample = nusc.get('sample', sample_token)
        except KeyError:
            # 跳过未知token（可能来自不同版本或裁剪集）
            continue

        scene_token = sample['scene_token']
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])
        ego_t = np.array(pose['translation'])
        ego_q = Quaternion(pose['rotation'])

        labels = pred['labels_3d'].cpu().numpy() if hasattr(pred['labels_3d'], 'cpu') else np.asarray(pred['labels_3d'])
        pts_3d = pred['pts_3d'].cpu().numpy() if hasattr(pred['pts_3d'], 'cpu') else np.asarray(pred['pts_3d'])
        if 'scores_3d' in pred:
            scores = pred['scores_3d'].cpu().numpy() if hasattr(pred['scores_3d'], 'cpu') else np.asarray(pred['scores_3d'])
        else:
            scores = np.ones((len(pts_3d),), dtype=np.float32)

        # MapTR 标签到名称映射（按项目实际设置，如需调整请修改）
        id_to_name = {0: 'divider', 1: 'ped_crossing', 2: 'boundary'}

        polys, types, keeps = [], [], []
        for i in range(len(pts_3d)):
            cls_name = id_to_name.get(int(labels[i]), 'unknown')
            if cls_name not in class_filter:
                continue
            if scores[i] < score_thr:
                continue
            polys.append(np.array(pts_3d[i], dtype=np.float32))
            types.append(cls_name)
            keeps.append(i)

        frame = {
            'token': sample_token,
            'timestamp': sample['timestamp'],
            'ego_translation': ego_t,
            'ego_rotation': ego_q,
            'polys': polys,
            'types': types,
            'scores': scores[keeps] if len(keeps) > 0 else np.zeros((0,), dtype=np.float32)
        }
        scene_to_frames.setdefault(scene_token, []).append(frame)

    # 按时间排序
    for st in list(scene_to_frames.keys()):
        scene_to_frames[st] = sorted(scene_to_frames[st], key=lambda x: x['timestamp'])

    return scene_to_frames


def transform_points_between_frames(points: np.ndarray,
                                   src_translation: np.ndarray,
                                   src_rotation: Quaternion,
                                   dst_translation: np.ndarray,
                                   dst_rotation: Quaternion) -> np.ndarray:
    """
    /**
     * @description 将二维点集从源帧自车坐标系变换到目标帧自车坐标系。
     * @param {np.ndarray} points (N,2) 点集
     * @param {np.ndarray} src_translation 源帧自车平移(x,y,z)
     * @param {Quaternion} src_rotation 源帧自车旋转
     * @param {np.ndarray} dst_translation 目标帧自车平移(x,y,z)
     * @param {Quaternion} dst_rotation 目标帧自车旋转
     * @returns {np.ndarray} (N,2) 目标帧坐标系下的点集
     */
    """
    src_translation = np.array(src_translation).flatten()
    dst_translation = np.array(dst_translation).flatten()

    if points.ndim == 1:
        points = points.reshape(1, -1)

    num_points = points.shape[0]
    points_3d = np.column_stack([points, np.zeros(num_points)])
    points_homo = np.column_stack([points_3d, np.ones(num_points)])

    src_to_world = np.eye(4, dtype=np.float64)
    src_to_world[:3, :3] = src_rotation.rotation_matrix
    src_to_world[:3, 3] = src_translation[:3]

    world_to_dst = np.eye(4, dtype=np.float64)
    world_to_dst[:3, :3] = dst_rotation.inverse.rotation_matrix
    world_to_dst[:3, 3] = -dst_rotation.inverse.rotate(dst_translation)

    transform = world_to_dst @ src_to_world
    transformed = (transform @ points_homo.T).T
    return transformed[:, :2]


def _apply_pred_local_adjust(points: np.ndarray,
                             rotate_deg: float = 0.0,
                             swap_xy: bool = False,
                             flip_x: bool = False,
                             flip_y: bool = False) -> np.ndarray:
    """
    /**
     * @description 在坐标变换前对预测点进行局部调整（旋转/交换/翻转）。
     * @param {np.ndarray} points (N,2) 点集
     * @param {number} rotate_deg 逆时针旋转角度（度）
     * @param {boolean} swap_xy 是否交换 x/y
     * @param {boolean} flip_x 是否翻转 x
     * @param {boolean} flip_y 是否翻转 y
     * @returns {np.ndarray} 调整后的点集
     */
    """
    if points is None or len(points) == 0:
        return points
    pts = np.array(points, dtype=np.float64)
    if swap_xy:
        pts = pts[:, [1, 0]]
    if flip_x:
        pts[:, 0] = -pts[:, 0]
    if flip_y:
        pts[:, 1] = -pts[:, 1]
    if abs(rotate_deg) > 1e-6:
        theta = np.deg2rad(rotate_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        pts = (R @ pts.T).T
    return pts


def transform_points_ego_to_world(points: np.ndarray,
                                  ego_translation: np.ndarray,
                                  ego_rotation: Quaternion) -> np.ndarray:
    """
    /**
     * @description 将二维点集从自车坐标系变换到世界坐标系。
     * @param {np.ndarray} points (N,2) 点集（自车系）
     * @param {np.ndarray} ego_translation 自车平移(x,y,z)
     * @param {Quaternion} ego_rotation 自车旋转
     * @returns {np.ndarray} (N,2) 世界坐标系下的点集
     */
    """
    world_zero_t = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    world_identity_q = Quaternion(1.0, 0.0, 0.0, 0.0)
    return transform_points_between_frames(points, ego_translation, ego_rotation,
                                           world_zero_t, world_identity_q)


def estimate_world_roi_from_egos(frames: List[dict], margin: float = 5.0) -> Tuple[float, float, float, float]:
    """
    /**
     * @description 基于一段场景内的自车轨迹估计紧凑的全局可视范围。
     * @param {Object[]} frames 有序帧列表
     * @param {number} margin 边界外扩（米）
     * @returns {number[]} [xmin, xmax, ymin, ymax]
     */
    """
    xs, ys = [], []
    for fr in frames:
        t = fr['ego_translation']
        xs.append(float(t[0]))
        ys.append(float(t[1]))
    if len(xs) == 0:
        return (-15.0, 15.0, -15.0, 15.0)  # 更紧凑的默认ROI
    xmin, xmax = float(np.min(xs)) - margin, float(np.max(xs)) + margin
    ymin, ymax = float(np.min(ys)) - margin, float(np.max(ys)) + margin
    return (xmin, xmax, ymin, ymax)


def estimate_world_roi_from_polys(frames: List[dict],
                                  margin: float = 2.0,
                                  rotate_deg: float = 0.0,
                                  swap_xy: bool = False,
                                  flip_x: bool = False,
                                  flip_y: bool = False,
                                  min_size: float = 20.0) -> Tuple[float, float, float, float]:
    """
    /**
     * @description 基于一段场景内的所有折线（变换到世界坐标系后）估计紧凑的全局可视范围。
     * @param {Object[]} frames 有序帧列表
     * @param {number} margin 边界外扩（米）
     * @param {number} rotate_deg 预测点局部旋转角度（度）
     * @param {boolean} swap_xy 是否交换 x/y
     * @param {boolean} flip_x 是否翻转 x
     * @param {boolean} flip_y 是否翻转 y
     * @param {number} min_size 最小ROI尺寸（米），确保不会过于狭小
     * @returns {number[]} [xmin, xmax, ymin, ymax]
     */
    """
    xs, ys = [], []
    for fr in frames:
        for p in fr.get('polys', []):
            if p is None or len(p) == 0:
                continue
            adj = _apply_pred_local_adjust(p, rotate_deg, swap_xy, flip_x, flip_y)
            wp = transform_points_ego_to_world(adj, fr['ego_translation'], fr['ego_rotation'])
            xs.extend(wp[:, 0].tolist())
            ys.extend(wp[:, 1].tolist())
    
    if len(xs) == 0:
        # 当没有有效折线时，基于自车轨迹创建紧凑ROI
        ego_xs, ego_ys = [], []
        for fr in frames:
            t = fr['ego_translation']
            ego_xs.append(float(t[0]))
            ego_ys.append(float(t[1]))
        if len(ego_xs) == 0:
            return (-min_size/2, min_size/2, -min_size/2, min_size/2)
        xmin, xmax = float(np.min(ego_xs)) - margin, float(np.max(ego_xs)) + margin
        ymin, ymax = float(np.min(ego_ys)) - margin, float(np.max(ego_ys)) + margin
    else:
        xmin, xmax = float(np.min(xs)) - margin, float(np.max(xs)) + margin
        ymin, ymax = float(np.min(ys)) - margin, float(np.max(ys)) + margin
    
    # 确保ROI不会过于狭小，保持最小尺寸
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    x_size, y_size = max(xmax - xmin, min_size), max(ymax - ymin, min_size)
    xmin, xmax = x_center - x_size/2, x_center + x_size/2
    ymin, ymax = y_center - y_size/2, y_center + y_size/2
    
    return (xmin, xmax, ymin, ymax)


def estimate_roi_from_overlay_window(frames_window: List[dict], 
                                   margin: float = 2.0, 
                                   min_size: float = 15.0) -> Tuple[float, float, float, float]:
    """
    /**
     * @description 基于当前窗口（历史帧已变换至当前帧自车坐标）内所有折线估计紧凑的自适应ROI。
     * @param {Object[]} frames_window 以当前帧为末尾的帧列表
     * @param {number} margin 边界外扩（米）
     * @param {number} min_size 最小ROI尺寸（米）
     * @returns {number[]} [xmin, xmax, ymin, ymax]
     */
    """
    if frames_window is None or len(frames_window) == 0:
        return (-min_size/2, min_size/2, -min_size/2, min_size/2)

    cur = frames_window[-1]
    xs, ys = [], []

    # 历史帧先变换到当前帧坐标
    for fr in frames_window[:-1]:
        for p in fr.get('polys', []):
            if p is None or len(p) == 0:
                continue
            tp = transform_points_between_frames(
                p,
                fr['ego_translation'], fr['ego_rotation'],
                cur['ego_translation'], cur['ego_rotation']
            )
            xs.extend(tp[:, 0].tolist())
            ys.extend(tp[:, 1].tolist())

    # 当前帧自身折线
    for p in cur.get('polys', []):
        if p is None or len(p) == 0:
            continue
        xs.extend(p[:, 0].tolist())
        ys.extend(p[:, 1].tolist())

    if len(xs) == 0:
        # 没有折线时，创建以自车为中心的紧凑ROI
        return (-min_size/2, min_size/2, -min_size/2, min_size/2)

    xmin, xmax = float(np.min(xs)) - margin, float(np.max(xs)) + margin
    ymin, ymax = float(np.min(ys)) - margin, float(np.max(ys)) + margin
    
    # 确保ROI不会过于狭小，保持最小尺寸
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    x_size, y_size = max(xmax - xmin, min_size), max(ymax - ymin, min_size)
    xmin, xmax = x_center - x_size/2, x_center + x_size/2
    ymin, ymax = y_center - y_size/2, y_center + y_size/2
    
    return (xmin, xmax, ymin, ymax)


def compute_optimal_roi(frames: List[dict], 
                       margin: float = 2.0,
                       min_size: float = 15.0,
                       rotate_deg: float = 0.0,
                       swap_xy: bool = False,
                       flip_x: bool = False,
                       flip_y: bool = False,
                       use_world_coords: bool = False) -> Tuple[float, float, float, float]:
    """
    /**
     * @description 计算最优的紧凑ROI，优先基于折线范围，备用自车轨迹。
     * @param {Object[]} frames 帧列表
     * @param {number} margin 边界外扩（米）
     * @param {number} min_size 最小ROI尺寸（米）
     * @param {number} rotate_deg 预测点局部旋转角度（度）
     * @param {boolean} swap_xy 是否交换 x/y
     * @param {boolean} flip_x 是否翻转 x
     * @param {boolean} flip_y 是否翻转 y
     * @param {boolean} use_world_coords 是否使用世界坐标系（否则使用最后一帧的自车坐标系）
     * @returns {number[]} [xmin, xmax, ymin, ymax]
     */
    """
    if not frames:
        return (-min_size/2, min_size/2, -min_size/2, min_size/2)
    
    xs, ys = [], []
    ref_frame = frames[-1] if not use_world_coords else None
    
    for fr in frames:
        for p in fr.get('polys', []):
            if p is None or len(p) == 0:
                continue
            
            adj = _apply_pred_local_adjust(p, rotate_deg, swap_xy, flip_x, flip_y)
            
            if use_world_coords:
                # 变换到世界坐标系
                coords = transform_points_ego_to_world(adj, fr['ego_translation'], fr['ego_rotation'])
            elif ref_frame is not None:
                # 变换到参考帧（最后一帧）的自车坐标系
                coords = transform_points_between_frames(
                    adj,
                    fr['ego_translation'], fr['ego_rotation'],
                    ref_frame['ego_translation'], ref_frame['ego_rotation']
                )
            else:
                # 直接使用自车坐标系
                coords = adj
            
            xs.extend(coords[:, 0].tolist())
            ys.extend(coords[:, 1].tolist())
    
    if len(xs) == 0:
        # 没有有效折线，基于自车轨迹
        if use_world_coords:
            for fr in frames:
                t = fr['ego_translation']
                xs.append(float(t[0]))
                ys.append(float(t[1]))
        else:
            # 在自车坐标系中，以原点为中心
            return (-min_size/2, min_size/2, -min_size/2, min_size/2)
    
    if len(xs) == 0:
        return (-min_size/2, min_size/2, -min_size/2, min_size/2)
    
    xmin, xmax = float(np.min(xs)) - margin, float(np.max(xs)) + margin
    ymin, ymax = float(np.min(ys)) - margin, float(np.max(ys)) + margin
    
    # 确保ROI不会过于狭小，保持最小尺寸
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    x_size, y_size = max(xmax - xmin, min_size), max(ymax - ymin, min_size)
    xmin, xmax = x_center - x_size/2, x_center + x_size/2
    ymin, ymax = y_center - y_size/2, y_center + y_size/2
    
    return (xmin, xmax, ymin, ymax)


def draw_single_frame_world(ax: plt.Axes,
                            frame: dict,
                            world_roi: Tuple[float, float, float, float],
                            line_width: float = 2.0,
                            rotate_deg: float = 0.0,
                            swap_xy: bool = False,
                            flip_x: bool = False,
                            flip_y: bool = False,
                            ego_alpha: float = 1.0) -> None:
    """
    /**
     * @description 在世界坐标系固定范围内，仅绘制当前帧的折线结果（不叠加历史）。
     * @param {Axes} ax Matplotlib Axes
     * @param {Object} frame 当前帧（含 ego 位姿与多段折线）
     * @param {number[]} world_roi [xmin, xmax, ymin, ymax]
     * @param {number} line_width 线宽
     * @param {float} ego_alpha 自车图标透明度
     */
    """
    xmin, xmax, ymin, ymax = world_roi
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # 绘制自车图标
    draw_ego_vehicle(ax, frame['ego_translation'], frame['ego_rotation'], alpha=ego_alpha)

    for p, t in zip(frame['polys'], frame['types']):
        if p is None or len(p) == 0:
            continue
        adj = _apply_pred_local_adjust(p, rotate_deg, swap_xy, flip_x, flip_y)
        print("自车坐标系下的坐标")
        print(adj)
        wp = transform_points_ego_to_world(adj, frame['ego_translation'], frame['ego_rotation'])
        print("世界坐标系下的坐标")
        print(wp)
        color = CLASS_TO_COLOR.get(t, (0.2, 0.2, 0.2))
        ax.plot(wp[:, 0], wp[:, 1], '-', color=color, alpha=1.0, linewidth=line_width)


def _get_world_polys_cached(frame: dict,
                            rotate_deg: float = 0.0,
                            swap_xy: bool = False,
                            flip_x: bool = False,
                            flip_y: bool = False) -> List[np.ndarray]:
    """
    /**
     * @description 读取或缓存当前帧折线的世界坐标表示。
     * @param {Object} frame 帧
     * @returns {np.ndarray[]} 世界坐标折线列表
     */
    """
    key = ('world_polys', rotate_deg, swap_xy, flip_x, flip_y)
    if key not in frame:
        world_polys = []
        for p in frame['polys']:
            if p is None or len(p) == 0:
                world_polys.append(p)
            else:
                adj = _apply_pred_local_adjust(p, rotate_deg, swap_xy, flip_x, flip_y)
                wp=transform_points_ego_to_world(adj, frame['ego_translation'], frame['ego_rotation'])
                world_polys.append(wp)
        frame[key] = world_polys
    return frame[key]


def draw_accumulated_world(ax: plt.Axes,
                           frames: List[dict],
                           world_roi: Tuple[float, float, float, float],
                           line_width: float = 2.0,
                           rotate_deg: float = 0.0,
                           swap_xy: bool = False,
                           flip_x: bool = False,
                           flip_y: bool = False) -> None:
    """
    /**
     * @description 在世界坐标系固定范围内，绘制从第一帧到当前帧的所有折线结果，形成累积地图。
     * @param {Axes} ax Matplotlib Axes
     * @param {Object[]} frames 从起始到当前的帧序列
     * @param {number[]} world_roi [xmin, xmax, ymin, ymax]
     * @param {number} line_width 线宽
     */
    """
    xmin, xmax, ymin, ymax = world_roi
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # 绘制自车轨迹，透明度逐帧增强
    for i, fr in enumerate(frames):
        # 计算透明度：从0.2逐渐增加到1.0
        alpha = 0.2 + (i / len(frames)) * 0.8
        draw_ego_vehicle(ax, fr['ego_translation'], fr['ego_rotation'], alpha=alpha)

    for fr in frames:
        world_polys = _get_world_polys_cached(fr, rotate_deg, swap_xy, flip_x, flip_y)
        for p, t in zip(world_polys, fr['types']):
            if p is None or len(p) == 0:
                continue
            color = CLASS_TO_COLOR.get(t, (0.2, 0.2, 0.2))
            ax.plot(p[:, 0], p[:, 1], '-', color=color, alpha=1.0, linewidth=line_width)


def poly_get_samples(poly: np.ndarray, num_samples: int = 100) -> np.ndarray:
    """
    /**
     * @description 生成沿 x 方向的等距采样点，用于折线近似比较。
     * @param {np.ndarray} poly (N,2) 折线点集
     * @param {number} num_samples 采样数
     * @returns {np.ndarray} (num_samples,) x 采样序列
     */
    """
    if poly is None or len(poly) == 0:
        return np.linspace(-1.0, 1.0, num_samples)
    x = [p[0] for p in poly]
    min_x, max_x = float(np.min(x)), float(np.max(x))
    return np.linspace(min_x, max_x, num_samples)


def _interp_poly_y(poly: np.ndarray, x_samples: np.ndarray) -> np.ndarray:
    if poly is None or len(poly) == 0:
        return np.zeros_like(x_samples)
    if len(poly) == 1:
        return np.full_like(x_samples, poly[0][1])

    y_samples = np.zeros_like(x_samples)
    segments = list(zip(poly[:-1], poly[1:]))
    current_segment = 0
    for i, x in enumerate(x_samples):
        while (current_segment < len(segments) and x > segments[current_segment][1][0]):
            current_segment += 1
        if current_segment >= len(segments):
            y_samples[i] = segments[-1][1][1]
        elif x < segments[current_segment][0][0]:
            y_samples[i] = segments[0][0][1]
        else:
            (x0, y0), (x1, y1) = segments[current_segment]
            if x1 == x0:
                y_samples[i] = (y0 + y1) / 2.0
            else:
                t = (x - x0) / (x1 - x0)
                y_samples[i] = y0 + t * (y1 - y0)
    return y_samples


def polyline_iou(poly1: np.ndarray, poly2: np.ndarray, x_samples: np.ndarray) -> float:
    """
    /**
     * @description 基于沿 x 方向的线性插值，计算两条折线的近似IoU相似度（0-1）。
     * @param {np.ndarray} poly1 (N,2)
     * @param {np.ndarray} poly2 (M,2)
     * @param {np.ndarray} x_samples 采样x坐标
     * @returns {number} IoU近似相似度 [0,1]
     */
    """
    if poly1 is None or poly2 is None or len(poly1) == 0 or len(poly2) == 0:
        return 0.0
    y1 = _interp_poly_y(poly1, x_samples)
    y2 = _interp_poly_y(poly2, x_samples)
    total_abs_diff = float(np.sum(np.abs(y1 - y2)))
    iou = 1.0 - total_abs_diff / (len(x_samples) * 15.0)
    return float(max(0.0, min(1.0, iou)))


def compute_temporal_consistency(cur_polys: List[np.ndarray], cur_types: List[str],
                                 prev_polys: List[np.ndarray]) -> Dict[str, float]:
    """
    /**
     * @description 计算每个类别的跨帧一致性：对每条当前折线，找上一帧中同类的最佳匹配IoU并取均值。
     * @param {np.ndarray[]} cur_polys 当前帧折线列表
     * @param {string[]} cur_types 对应类别名称列表
     * @param {np.ndarray[]} prev_polys 上一帧(已变换到当前帧坐标系)的折线列表（与 cur_types 同长度并同类过滤）
     * @returns {Object} 类别 -> 一致性均值 [0,1]
     */
    """
    by_cls_cur: Dict[str, List[np.ndarray]] = {c: [] for c in CLASS_NAMES}
    for p, t in zip(cur_polys, cur_types):
        if t in by_cls_cur:
            by_cls_cur[t].append(p)

    # 为简化，这里假设 prev_polys 中的类别分布与 cur_types 相同次序（绘图时我们将同步过滤）
    # 实际生产中应同时传入 prev_types 并分别分桶。
    consistency: Dict[str, float] = {}
    for cls in CLASS_NAMES:
        cur_list = by_cls_cur.get(cls, [])
        if len(cur_list) == 0 or len(prev_polys) == 0:
            consistency[cls] = 0.0
            continue
        scores = []
        for cp in cur_list:
            xs = poly_get_samples(cp, 100)
            # 与上一帧所有折线（不区分类）比较：近似取最大值（若需严格同类，可传 prev_types 并过滤）
            best = 0.0
            for pp in prev_polys:
                best = max(best, polyline_iou(cp, pp, xs))
            scores.append(best)
        consistency[cls] = float(np.mean(scores)) if scores else 0.0
    return consistency


def draw_overlay(ax: plt.Axes,
                 frames: List[dict],
                 window: int,
                 roi: Tuple[float, float, float, float],
                 alpha_decay: float = 0.65,
                 line_width: float = 2.0) -> Dict[str, float]:
    """
    /**
     * @description 绘制当前帧与前 window-1 帧（已变换）的矢量叠加，返回一致性评分。
     * @param {Axes} ax Matplotlib Axes
     * @param {Object[]} frames 以当前帧为末尾的帧列表，长度≤window
     * @param {number[]} roi [xmin, xmax, ymin, ymax]
     * @param {number} alpha_decay 透明度衰减（0-1，越小衰减越快）
     * @param {number} line_width 线宽
     * @returns {Object} 每类一致性评分
     */
    """
    xmin, xmax, ymin, ymax = roi
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    cur = frames[-1]
    cur_polys = cur['polys']
    cur_types = cur['types']

    # 绘制自车轨迹，透明度逐帧增强
    for i, fr in enumerate(frames):
        # 计算透明度：从0.2逐渐增加到1.0
        alpha = 0.2 + (i / len(frames)) * 0.8
        # 将历史帧的自车位置变换到当前帧坐标系
        if i < len(frames) - 1:
            # 历史帧：变换到当前帧坐标系
            ego_pos_2d = np.array([0, 0])  # 自车在自车坐标系中的位置
            transformed_ego_pos = transform_points_between_frames(
                ego_pos_2d,
                fr['ego_translation'], fr['ego_rotation'],
                cur['ego_translation'], cur['ego_rotation']
            )
            # 创建临时的自车位置和旋转信息
            temp_ego_translation = np.array([transformed_ego_pos[0], transformed_ego_pos[1], 0])
            temp_ego_rotation = Quaternion(axis=[0, 0, 1], angle=0)  # 简化处理
            draw_ego_vehicle(ax, temp_ego_translation, temp_ego_rotation, alpha=alpha)
        else:
            # 当前帧：直接绘制
            draw_ego_vehicle(ax, cur['ego_translation'], cur['ego_rotation'], alpha=alpha)

    # 收集上一帧（合并）用于一致性估计
    prev_merged_polys: List[np.ndarray] = []

    # 先画历史帧（从老到新），透明度递增
    for i, fr in enumerate(frames[:-1]):
        # 变换到当前帧
        alpha = (alpha_decay ** (len(frames) - 1 - i))
        for p, t in zip(fr['polys'], fr['types']):
            if len(p) == 0:
                continue
            tp = transform_points_between_frames(
                p,
                fr['ego_translation'], fr['ego_rotation'],
                cur['ego_translation'], cur['ego_rotation']
            )
            prev_merged_polys.append(tp)
            color = CLASS_TO_COLOR.get(t, (0.6, 0.6, 0.6))
            ax.plot(tp[:, 0], tp[:, 1], '-', color=color, alpha=alpha, linewidth=line_width)

    # 再画当前帧（更醒目）
    for p, t in zip(cur_polys, cur_types):
        if len(p) == 0:
            continue
        color = CLASS_TO_COLOR.get(t, (0.2, 0.2, 0.2))
        ax.plot(p[:, 0], p[:, 1], '-', color=color, alpha=1.0, linewidth=line_width)

    # 计算并返回一致性（简单近邻IoU）
    consistency = compute_temporal_consistency(cur_polys, cur_types, prev_merged_polys)
    return consistency


def save_frame(fig: plt.Figure, save_dir: str, idx: int) -> str:
    """
    /**
     * @description 保存单帧图像
     * @param {Figure} fig Matplotlib Figure
     * @param {str} save_dir 输出目录
     * @param {number} idx 序号
     * @returns {str} 文件路径
     */
    """
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, f'{idx:06d}.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def maybe_write_gif(img_paths: List[str], out_path: str, fps: int = 6) -> None:
    """
    /**
     * @description 将已保存的帧合成为GIF
     * @param {string[]} img_paths 图像路径序列
     * @param {str} out_path 输出gif路径
     * @param {number} fps 帧率
     */
    """
    try:
        import imageio.v2 as imageio
        # 读取并规范化通道为RGB，必要时对尺寸做白色填充以对齐
        imgs = []
        for p in img_paths:
            arr = imageio.imread(p)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            imgs.append(arr)

        # 统一尺寸（避免 tight bbox 或渲染差异导致的像素尺寸不一致）
        max_h = max(im.shape[0] for im in imgs)
        max_w = max(im.shape[1] for im in imgs)
        norm_imgs = []
        for im in imgs:
            h, w = im.shape[:2]
            if h != max_h or w != max_w:
                canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
                canvas[:h, :w, :3] = im.astype(np.uint8)
                norm_imgs.append(canvas)
            else:
                norm_imgs.append(im.astype(np.uint8))

        imageio.mimsave(out_path, norm_imgs, fps=fps)
    except Exception as e:
        print(f'[warn] 生成GIF失败: {e}')


def _safe_load_image_rgb(path: str) -> np.ndarray:
    """
    /**
     * @description 安全读取图像为RGB三通道
     * @param {str} path 图像路径
     * @returns {np.ndarray} (H,W,3) uint8 图像
     */
    """
    try:
        import imageio.v2 as imageio
        arr = imageio.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = arr.astype(np.uint8)
        return arr
    except Exception:
        # 读取失败则返回一个小白图
        return np.full((10, 10, 3), 255, dtype=np.uint8)


def _six_view_mosaic(nusc: NuScenes, sample_token: str, dataroot: str) -> np.ndarray:
    """
    /**
     * @description 生成六视图(2x3)马赛克图像，按行从左到右：FRONT_LEFT, FRONT, FRONT_RIGHT / BACK_LEFT, BACK, BACK_RIGHT
     * @param {NuScenes} nusc NuScenes 句柄
     * @param {str} sample_token sample token
     * @param {str} dataroot 数据根目录
     * @returns {np.ndarray} (H,W,3) uint8 马赛克图
     */
    """
    channels = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    try:
        sample = nusc.get('sample', sample_token)
    except Exception:
        return np.full((10, 10, 3), 255, dtype=np.uint8)

    imgs: List[np.ndarray] = []
    for ch in channels:
        sd_token = sample['data'].get(ch, None)
        if sd_token is None:
            imgs.append(np.full((10, 10, 3), 255, dtype=np.uint8))
            continue
        sd = nusc.get('sample_data', sd_token)
        img_path = os.path.join(dataroot, sd['filename'])
        imgs.append(_safe_load_image_rgb(img_path))

    # 对齐每行内的高度，按最小高度裁剪或填充到行内最小高度
    row1 = imgs[0:3]
    row2 = imgs[3:6]
    h1 = min(im.shape[0] for im in row1)
    h2 = min(im.shape[0] for im in row2)

    def _h_crop(im: np.ndarray, h: int) -> np.ndarray:
        if im.shape[0] == h:
            return im
        if im.shape[0] > h:
            return im[:h, :, :]
        # 填充至目标高度（白色）
        pad = np.full((h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([im, pad], axis=0)

    row1 = [_h_crop(im, h1) for im in row1]
    row2 = [_h_crop(im, h2) for im in row2]

    row1_cat = np.concatenate(row1, axis=1)
    row2_cat = np.concatenate(row2, axis=1)

    # 纵向对齐到相同宽度
    w = max(row1_cat.shape[1], row2_cat.shape[1])
    def _w_pad(im: np.ndarray, w_target: int) -> np.ndarray:
        if im.shape[1] == w_target:
            return im
        pad = np.full((im.shape[0], w_target - im.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([im, pad], axis=1)

    row1_cat = _w_pad(row1_cat, w)
    row2_cat = _w_pad(row2_cat, w)

    mosaic = np.concatenate([row1_cat, row2_cat], axis=0)
    return mosaic







def parse_args():
    """
    /**
     * @description 解析命令行参数
     */
    """
    parser = argparse.ArgumentParser('MapTR 连续帧稳定性可视化')
    parser.add_argument('--pred', type=str, default=None, help='预测输出pkl路径 (test脚本 --out 保存)，不指定时可视化真值')
    parser.add_argument('--data-root', type=str, required=True, help='NuScenes数据路径')
    parser.add_argument('--gt-path', type=str, default='data/nuscenes/nuscenes_map_infos_temporal_val.pkl', help='真值数据pkl路径')
    parser.add_argument('--nusc-version', type=str, default='v1.0-mini', help='NuScenes版本')
    parser.add_argument('--out-dir', type=str, default='./vis_stability', help='输出目录')
    parser.add_argument('--scene-token', type=str, default=None, help='指定scene_token（缺省时可视化第一段）')
    parser.add_argument('--scene-name', type=str, default=None, help='指定scene名称（如 scene-0103）')
    parser.add_argument('--scene-index', type=int, default=None, help='指定scene索引（基于NuScenes顺序，从0开始）')
    parser.add_argument('--num-frames', type=int, default=10, help='导出帧数上限（从段起始计）')
    parser.add_argument('--interval', type=int, default=2, help='可视窗口步长（相邻帧间隔）')
    parser.add_argument('--window', type=int, default=5, help='叠加可视的窗口帧数')
    parser.add_argument('--roi', type=float, nargs=4, default=[-25.0, 25.0, -25.0, 25.0], help='可视范围 [xmin xmax ymin ymax]')
    parser.add_argument('--roi-margin', type=float, default=2.0, help='动态ROI外扩（米）')
    parser.add_argument('--alpha-decay', type=float, default=0.7, help='历史帧透明度衰减系数(0-1)')
    parser.add_argument('--line-width', type=float, default=1.0, help='线宽')
    parser.add_argument('--score-thr', type=float, default=0.3, help='分数阈值')
    parser.add_argument('--classes', type=str, nargs='+', default=CLASS_NAMES, help='可视类别')
    parser.add_argument('--gif', action='store_true', help='是否导出GIF')
    parser.add_argument('--gif-fps', type=int, default=6, help='GIF帧率')
    parser.add_argument('--gif-world-margin', type=float, default=2.0, help='GIF世界视图边界外扩（米）')
    parser.add_argument('--no-gif-accumulate', action='store_true', help='关闭 GIF 世界视图逐帧累积（默认累积开启）')
    parser.add_argument('--gif-separate', action='store_true', default=True, help='分别生成六视图GIF和地图GIF（默认启用）')
    parser.add_argument('--gif-combined', action='store_true', help='生成包含六视图和地图的组合GIF（不推荐）')
    # 预测坐标系调整（在 ego->world 变换之前先应用）
    parser.add_argument('--pred-rotate-deg', type=float, default=0.0, help='对预测点先旋转角度（度, 绕Z）')
    parser.add_argument('--pred-swap-xy', action='store_true', help='对预测点先交换 x/y')
    parser.add_argument('--pred-flip-x', action='store_true', help='对预测点先翻转 x')
    parser.add_argument('--pred-flip-y', action='store_true', help='对预测点先翻转 y')
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # 确定使用的数据路径和类型
    if args.pred is not None:
        data_path = args.pred
        data_type = "预测"
        pkl_name = args.pred.split('/')[-1].split('.')[0]
    else:
        data_path = args.gt_path
        data_type = "真值"
        pkl_name = "gt_" + args.gt_path.split('/')[-1].split('.')[0]
    
    base_out = os.path.join(args.out_dir, f'stable_vis_{pkl_name}/{args.scene_name}')
    _ensure_dir(base_out)

    print(f'加载 NuScenes 与{data_type}结果...')
    nusc = NuScenes(version=args.nusc_version, dataroot=args.data_root, verbose=False)
    
    # 加载数据
    if args.pred is not None:
        outputs = _load_outputs(args.pred)
        print(f'加载预测结果: {args.pred}')
    else:
        gt_data = _load_outputs(args.gt_path)
        outputs = _convert_gt_to_outputs_format(gt_data)
        print(f'加载真值结果: {args.gt_path}')

    print('整理场景序列...')
    scene_to_frames = _collect_scene_sequences(nusc, outputs, score_thr=args.score_thr, class_filter=args.classes)
    if len(scene_to_frames) == 0:
        print(f'[warn] 没有可视化的场景/帧，请检查 {data_type}数据路径, --data-root 与类别过滤')
        return

    # 选择场景：优先 token -> name -> index -> 第一个
    chosen_scene = None
    if args.scene_token:
        if args.scene_token in scene_to_frames:
            chosen_scene = args.scene_token
        else:
            print(f"[warn] 指定的 scene_token 不在{data_type}数据中: {args.scene_token}")

    if chosen_scene is None and args.scene_name:
        # 遍历NuScenes的scene表以匹配name到token
        try:
            for sc in nusc.scene:
                if sc.get('name') == args.scene_name:
                    maybe_token = sc.get('token')
                    if maybe_token in scene_to_frames:
                        chosen_scene = maybe_token
                        break
            if chosen_scene is None:
                print(f"[warn] 指定的 scene-name 未在{data_type}数据中找到: {args.scene_name}")
        except Exception as e:
            print(f"[warn] 解析 scene-name 失败: {e}")

    if chosen_scene is None and args.scene_index is not None:
        try:
            idx = int(args.scene_index)
            if 0 <= idx < len(nusc.scene):
                maybe_token = nusc.scene[idx]['token']
                if maybe_token in scene_to_frames:
                    chosen_scene = maybe_token
                else:
                    print(f"[warn] scene-index 对应的场景未在{data_type}数据中: index={idx}")
            else:
                print(f"[warn] scene-index 越界: {idx}, 合法范围 [0, {len(nusc.scene)-1}]")
        except Exception as e:
            print(f"[warn] 解析 scene-index 失败: {e}")

    if chosen_scene is None:
        chosen_scene = list(scene_to_frames.keys())[0]
        print(f"[info] 未明确指定或未匹配到场景，使用第一个{data_type}场景: {chosen_scene}")

    frames = scene_to_frames[chosen_scene]
    if len(frames) == 0:
        print(f'[warn] 该场景无{data_type}帧数据')
        return

    print(f'开始绘制场景 {chosen_scene}，{data_type}帧数: {len(frames)}')
    save_dir = os.path.join(base_out, chosen_scene)
    _ensure_dir(save_dir)

    # 根据是否分别生成GIF，初始化不同的路径列表
    if args.gif and args.gif_separate:
        map_img_paths: List[str] = []
        six_img_paths: List[str] = []
        save_dir_map = os.path.join(save_dir, 'map_frames')
        save_dir_six = os.path.join(save_dir, 'six_view_frames')
        _ensure_dir(save_dir_map)
        _ensure_dir(save_dir_six)
    else:
        img_paths: List[str] = []

    roi_default = [-25.0, 25.0, -25.0, 25.0]
    window = max(1, int(args.window))
    step = max(1, int(args.interval))
    max_frames = min(int(args.num_frames), len(frames))

    # 若导出 GIF，则按帧动态计算世界坐标系 ROI，避免整体范围过大导致空白
    world_roi = None

    start_idx = 0 if args.gif else (window - 1)
    for idx in range(start_idx, max_frames, step):
        if args.gif:
            # 世界范围：根据是否累积选择 ROI 计算的帧集合
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1)
            if args.no_gif_accumulate:
                sub_frames = [frames[idx]]
            else:
                sub_frames = frames[:idx+1]
            world_roi = compute_optimal_roi(
                sub_frames,
                margin=float(args.gif_world_margin),
                rotate_deg=float(args.pred_rotate_deg),
                swap_xy=bool(args.pred_swap_xy),
                flip_x=bool(args.pred_flip_x),
                flip_y=bool(args.pred_flip_y),
                use_world_coords=True
            )
            if args.no_gif_accumulate:
                fr = frames[idx]
                # 计算自车透明度：当前帧为1.0
                ego_alpha = 1.0
                draw_single_frame_world(ax, fr, world_roi,
                                        line_width=float(args.line_width),
                                        rotate_deg=float(args.pred_rotate_deg),
                                        swap_xy=bool(args.pred_swap_xy),
                                        flip_x=bool(args.pred_flip_x),
                                        flip_y=bool(args.pred_flip_y),
                                        ego_alpha=ego_alpha)
                ts = fr['timestamp']
                ax.set_title(f't={ts}  World-View', fontsize=10)
            else:
                draw_accumulated_world(
                    ax, frames[:idx+1], world_roi,
                    line_width=float(args.line_width),
                    rotate_deg=float(args.pred_rotate_deg),
                    swap_xy=bool(args.pred_swap_xy),
                    flip_x=bool(args.pred_flip_x),
                    flip_y=bool(args.pred_flip_y)
                )
                ts = frames[idx]['timestamp']
                ax.set_title(f't={ts}  World-Accumulated', fontsize=10)
        else:
            # 保持原本稳定性叠加视图
            sub = frames[max(0, idx - (window - 1)): idx + 1]
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1)

            # 自适应或固定 ROI：若用户自定义了 --roi（非默认值）则使用固定，否则基于窗口动态估计
            if list(args.roi) != roi_default:
                cur_roi = tuple(args.roi)
            else:
                cur_roi = compute_optimal_roi(sub, margin=float(args.roi_margin), use_world_coords=False)

            consistency = draw_overlay(ax, sub, window=window, roi=cur_roi,
                                       alpha_decay=float(args.alpha_decay),
                                       line_width=float(args.line_width))

            # 标题：帧时间与一致性
            ts = sub[-1]['timestamp']
            cons_str = ' | '.join([f"{k}:{consistency.get(k, 0.0):.2f}" for k in CLASS_NAMES])
            ax.set_title(f't={ts}  Consistency({window}): {cons_str}', fontsize=10)

        # 保存可视化帧
        if args.gif:
            # 生成地图帧
            map_vis_path = save_frame(fig, save_dir_map, idx)
            map_img_paths.append(map_vis_path)

            # 生成六视图
            if args.no_gif_accumulate:
                st = frames[idx]['token']
            else:
                st = frames[idx]['token']

            six_fig = plt.figure(figsize=(8, 6))
            six_ax = six_fig.add_subplot(1, 1, 1)
            six_img = _six_view_mosaic(nusc, st, args.data_root)

            # 显示六视图
            six_ax.imshow(six_img)
            six_ax.set_title(f't={frames[idx]["timestamp"]}  Six-View', fontsize=10)
            six_ax.axis('off')

            six_vis_path = save_frame(six_fig, save_dir_six, idx)
            six_img_paths.append(six_vis_path)

    # 生成GIF
    if args.gif:
        # 生成地图GIF
        if len(map_img_paths) > 0:
            map_gif_path = os.path.join(base_out, f'{chosen_scene}_map.gif')
            print(f'生成地图GIF: {map_gif_path}')
            maybe_write_gif(map_img_paths, map_gif_path, fps=int(args.gif_fps))

        # 生成六视图GIF
        if len(six_img_paths) > 0:
            six_gif_path = os.path.join(base_out, f'{chosen_scene}_six_view.gif')
            print(f'生成六视图GIF: {six_gif_path}')
            maybe_write_gif(six_img_paths, six_gif_path, fps=int(args.gif_fps))

        print(f'地图帧保存在: {save_dir_map}')
        print(f'六视图帧保存在: {save_dir_six}')

    print(f'完成，可视化结果保存在: {save_dir}')


if __name__ == '__main__':
    main()


