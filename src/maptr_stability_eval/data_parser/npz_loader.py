"""
NPZ文件加载和数据解析模块

支持加载PivotNet等模型输出的npz格式文件，每个npz文件对应一个token。
适用于文件夹中包含多个npz文件的场景。
"""

import os
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import torch

from ..geometry import _apply_pred_local_adjust


def _clean_and_pad_polylines(polylines_list, *, token: str = None, debug: bool = False, config: dict = None):
    """
    清洗并填充折线到统一长度（使用末点重复而非零填充）
    
    Args:
        polylines_list: list[np.ndarray]，每个元素为 (Pi, 2) 的折线点集合
    
    Returns:
        padded_pts_3d: np.ndarray，形状 (N, Pmax, 2)，使用末点重复进行填充
        valid_index: list[int]，保留下来的折线在输入列表中的索引
    """
    cleaned_polylines = []
    valid_index = []
    removed_count = 0
    for idx, poly in enumerate(polylines_list):
        if poly is None:
            removed_count += 1
            continue
        pts = np.asarray(poly)
        # 容忍 (N,3/4) 等多列，只取前两列
        if pts.ndim == 1 and pts.size >= 2:
            pts = np.asarray(pts[:2], dtype=np.float32).reshape(1, 2)
        elif pts.ndim >= 2 and pts.shape[-1] >= 2:
            pts = np.asarray(pts[..., :2], dtype=np.float32)
        else:
            removed_count += 1
            continue
        # 仅保留有限值点
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.shape[0] == 0:
            removed_count += 1
            continue
        cleaned_polylines.append(pts)
        valid_index.append(idx)

    if len(cleaned_polylines) == 0:
        return np.zeros((0, 0, 2), dtype=np.float32), []

    max_points = max(len(p) for p in cleaned_polylines)
    num_polylines = len(cleaned_polylines)
    padded_pts_3d = np.zeros((num_polylines, max_points, 2), dtype=np.float32)
    for i, pts in enumerate(cleaned_polylines):
        n = pts.shape[0]
        padded_pts_3d[i, :n] = pts
        if n < max_points:
            # 末点重复填充，避免引入(0,0)假点影响定位稳定性
            padded_pts_3d[i, n:max_points] = pts[-1]
    if debug and token is not None:
        total = len(polylines_list)
        kept = len(cleaned_polylines)
        # 统计窗口内比例（在启用auto_adjust前）。窗口来自配置 data_processing.x_range/y_range；默认 x在[-15,15]，y在[-30,30]
        flat = padded_pts_3d.reshape(-1, 2)
        x_range = (-15.0, 15.0)
        y_range = (-30.0, 30.0)
        try:
            if isinstance(config, dict):
                dp = config.get('data_processing', {})
                if 'x_range' in dp:
                    xr = dp['x_range']
                    if isinstance(xr, (list, tuple)) and len(xr) == 2:
                        x_range = (float(xr[0]), float(xr[1]))
                if 'y_range' in dp:
                    yr = dp['y_range']
                    if isinstance(yr, (list, tuple)) and len(yr) == 2:
                        y_range = (float(yr[0]), float(yr[1]))
        except Exception:
            pass
        x_abs = max(abs(x_range[0]), abs(x_range[1]))
        y_abs = max(abs(y_range[0]), abs(y_range[1]))
        ax = np.abs(flat)
        inside = float(np.mean((ax[:, 0] <= x_abs) & (ax[:, 1] <= y_abs))) if flat.size else 0.0
        print(f"[npz_loader][debug] token={token} 折线清洗: 保留 {kept}/{total}, 移除 {removed_count}, inside_ratio_pre_adjust={inside:.3f}")
    return padded_pts_3d, valid_index


def _to_numpy(x):
    """将输入转换为 numpy.ndarray，支持 torch.Tensor（含CUDA）、list、tuple。"""
    try:
        import numpy as _np
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().numpy()
        return _np.asarray(x)
    except Exception:
        try:
            return np.asarray(x)
        except Exception:
            return None


def _to_python_str(value):
    """尽可能将numpy/bytes等字符串表示转换为Python str。"""
    try:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            elif value.size == 1:
                value = value.reshape(-1)[0]
            else:
                return None
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        if value is None:
            return None
        return str(value)
    except Exception:
        return None


def _detect_bev_pixel_coords(padded_pts_3d, width: int, height: int) -> bool:
    """简单启发式：若大部分点都在[0,width]x[0,height]范围内，则视为BEV像素坐标。"""
    try:
        pts = padded_pts_3d.reshape(-1, 2)
        if pts.size == 0:
            return False
        inside = (pts[:, 0] >= 0) & (pts[:, 0] <= (width + 1)) & (pts[:, 1] >= 0) & (pts[:, 1] <= (height + 1))
        ratio = float(np.mean(inside))
        return ratio >= 0.8
    except Exception:
        return False


def _bev_pixels_to_ego_meters(padded_pts_3d, width: int, height: int, *, radius_x_m: float = 50.0, radius_y_m: float = None):
    """
    将BEV像素坐标(u,v)转换为以自车为中心的米制坐标(x,y)。
    - 像素坐标原点假定在左上角，u向右增大，v向下增大。
    - 转换后原点在图像中心，x向右(车辆左/右视具体定义)，y向上；再按米制尺度放大。
    - 半径：width 对应 2*radius_x_m，height 对应 2*radius_y_m（若未显式给出，则按宽高比缩放）。
    """
    pts = padded_pts_3d.astype(np.float32)
    if radius_y_m is None:
        # 依据宽高比缩放（如400x200 -> 50m x 25m）
        radius_y_m = radius_x_m * (float(height) / float(width))
    # 像素中心坐标
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    scale_x = (2.0 * radius_x_m) / max(float(width) - 1.0, 1.0)
    scale_y = (2.0 * radius_y_m) / max(float(height) - 1.0, 1.0)
    
    # 逐折线转换：在扁平视图计算，避免形状广播错误
    flat_in = pts.reshape(-1, 2)
    out_flat = np.empty_like(flat_in)
    # u,v -> 以中心为原点，并翻转v方向
    # print("===============================bev_pixels_to_ego_meters1===========================")
    out_flat[:, 0] = (flat_in[:, 0] - cx) * scale_x
    # print("===============================bev_pixels_to_ego_meters2===========================")
    out_flat[:, 1] = (cy - flat_in[:, 1]) * scale_y
    return out_flat.reshape(padded_pts_3d.shape).astype(np.float32)


def _quat_to_rot2_wxyz(q):
    """四元数[w,x,y,z] 转 2x2 旋转矩阵（取3x3的左上角）。"""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size == 4:
        w, x, y, z = q[0], q[1], q[2], q[3]
    elif q.size == 3:
        # 退化情形，按[yaw]近似
        cy, sy = np.cos(q[2]), np.sin(q[2])
        return np.array([[cy, -sy], [sy, cy]], dtype=np.float64)
    else:
        return np.eye(2, dtype=np.float64)
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R[:2, :2]


def _quat_to_rot2_xyzw(q):
    """四元数[x,y,z,w] 转 2x2 旋转矩阵（取3x3的左上角）。"""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size == 4:
        x, y, z, w = q[0], q[1], q[2], q[3]
    else:
        return np.eye(2, dtype=np.float64)
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R[:2, :2]


def _transform_global_to_ego_batch(pts_np, t_np, q_np):
    """尝试两种四元数顺序（wxyz/xyzw）进行全局->自车转换，选择更合理的结果。"""
    pts = pts_np.reshape(-1, 2).astype(np.float64)
    t = np.asarray(t_np, dtype=np.float64).reshape(-1)
    if t.size >= 2:
        pts_centered = pts - t[:2]
    else:
        pts_centered = pts

    # 候选1：wxyz
    R1 = _quat_to_rot2_wxyz(q_np)
    cand1 = (R1.T @ pts_centered.T).T
    # 候选2：xyzw
    R2 = _quat_to_rot2_xyzw(q_np)
    cand2 = (R2.T @ pts_centered.T).T

    # 选择落入局部窗口[-100,100]比例更高者
    def score(arr):
        if arr.size == 0:
            return -1
        ax = np.abs(arr)
        inside = np.logical_and(ax[:, 0] <= 100.0, ax[:, 1] <= 100.0)
        return float(np.mean(inside))

    s1, s2 = score(cand1), score(cand2)
    chosen = cand1 if s1 >= s2 else cand2
    return chosen.reshape(pts_np.shape).astype(np.float32)


def _maybe_transform_to_ego(padded_pts_3d, ego_translation, ego_rotation, config):
    """
    若输入为全局坐标，则转换为自车坐标。启用条件：
    - 配置中显式指定 assume_global_coords=True；或
    - 自动检测到坐标量级异常（最大绝对值>200）。
    """
    if padded_pts_3d.size == 0:
        return padded_pts_3d

    assume_global = bool(config.get('assume_global_coords', False))
    if not assume_global:
        # 简单量级检测：局部坐标通常在百米级以内
        max_abs = float(np.nanmax(np.abs(padded_pts_3d)))
        if max_abs > 200.0:
            assume_global = True

    if not assume_global:
        return padded_pts_3d

    if ego_translation is None or ego_rotation is None:
        return padded_pts_3d

    try:
        t_np = _to_numpy(ego_translation)
        q_np = _to_numpy(ego_rotation)
        if t_np is None or q_np is None:
            return padded_pts_3d
        return _transform_global_to_ego_batch(padded_pts_3d, t_np, q_np)
    except Exception:
        return padded_pts_3d


def _auto_adjust_orientation(padded_pts_3d, config, *, token=None):
    """
    自动选择 swap/flip/rotate 组合，使更多点落入评估窗口（默认 x∈[-15,15], y∈[-30,30]，或取自配置 data_processing.x_range/y_range）。
    仅在 config['auto_adjust_orientation'] 为真时启用。
    """
    if not bool(config.get('auto_adjust_orientation', False)):
        return padded_pts_3d
    if padded_pts_3d.size == 0:
        return padded_pts_3d

    def inside_score(arr):
        if arr.size == 0:
            return -1.0
        # 依据配置的窗口统计 inside 比例
        x_range = (-15.0, 15.0)
        y_range = (-30.0, 30.0)
        try:
            dp = config.get('data_processing', {}) if isinstance(config, dict) else {}
            xr = dp.get('x_range', x_range)
            yr = dp.get('y_range', y_range)
            if isinstance(xr, (list, tuple)) and len(xr) == 2:
                x_range = (float(xr[0]), float(xr[1]))
            if isinstance(yr, (list, tuple)) and len(yr) == 2:
                y_range = (float(yr[0]), float(yr[1]))
        except Exception:
            pass
        x_abs = max(abs(x_range[0]), abs(x_range[1]))
        y_abs = max(abs(y_range[0]), abs(y_range[1]))
        ax = np.abs(arr.reshape(-1, 2))
        return float(np.mean((ax[:, 0] <= x_abs) & (ax[:, 1] <= y_abs)))

    candidates_rotate = [0.0, 90.0, -90.0, 180.0]
    candidates_swap = [False, True]
    candidates_flipx = [False, True]
    candidates_flipy = [False, True]

    best = None
    pts_best = padded_pts_3d
    flat = padded_pts_3d.reshape(-1, 2)

    for rot in candidates_rotate:
        for sw in candidates_swap:
            for fx in candidates_flipx:
                for fy in candidates_flipy:
                    pts_list = []
                    # 逐折线应用，避免函数内部复制形状造成问题
                    for poly in padded_pts_3d:
                        pts_list.append(_apply_pred_local_adjust(poly, rotate_deg=rot, swap_xy=sw, flip_x=fx, flip_y=fy))
                    cand = np.stack(pts_list, axis=0).astype(np.float32)
                    sc = inside_score(cand)
                    if (best is None) or (sc > best[0]):
                        best = (sc, rot, sw, fx, fy)
                        pts_best = cand

    if token is not None and bool(config.get('debug_npz_loader', False)):
        sc, rot, sw, fx, fy = best
        flat = pts_best.reshape(-1, 2)
        # 复用上面的窗口设置进行统计
        x_range = (-15.0, 15.0)
        y_range = (-30.0, 30.0)
        try:
            dp = config.get('data_processing', {}) if isinstance(config, dict) else {}
            xr = dp.get('x_range', x_range)
            yr = dp.get('y_range', y_range)
            if isinstance(xr, (list, tuple)) and len(xr) == 2:
                x_range = (float(xr[0]), float(xr[1]))
            if isinstance(yr, (list, tuple)) and len(yr) == 2:
                y_range = (float(yr[0]), float(yr[1]))
        except Exception:
            pass
        x_abs = max(abs(x_range[0]), abs(x_range[1]))
        y_abs = max(abs(y_range[0]), abs(y_range[1]))
        ax = np.abs(flat)
        inside = float(np.mean((ax[:, 0] <= x_abs) & (ax[:, 1] <= y_abs))) if flat.size else 0.0
        print(f"[npz_loader][debug] token={token} auto_adjust chosen: score={sc:.3f}, rot={rot}, swap={sw}, flip_x={fx}, flip_y={fy}, inside_ratio_post_adjust={inside:.3f}")

    return pts_best


def load_npz_prediction_results(npz_dir, config):
    """
    从npz文件夹加载预测结果
    
    Args:
        npz_dir: npz文件所在目录路径
        config: 配置字典，包含字段映射信息
        
    Returns:
        prediction_results: 预测结果列表，格式与pkl加载器保持一致
    """
    print(f"加载npz预测结果目录: {npz_dir}")
    
    if not os.path.exists(npz_dir):
        raise ValueError(f"npz目录不存在: {npz_dir}")
    
    if not os.path.isdir(npz_dir):
        raise ValueError(f"路径不是目录: {npz_dir}")
    
    # 查找所有npz文件
    npz_pattern = os.path.join(npz_dir, "*.npz")
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        raise ValueError(f"在目录 {npz_dir} 中未找到npz文件")
    
    print(f"找到 {len(npz_files)} 个npz文件")
    
    # 加载所有npz文件
    prediction_results = []
    failed_files = []
    
    for npz_file in tqdm(npz_files, desc="加载npz文件"):
        try:
            # 从文件名提取token（去掉.npz扩展名）
            token = Path(npz_file).stem
            
            # 加载npz文件
            npz_data = np.load(npz_file, allow_pickle=True)
            # print(npz_data)

            # print(npz_data)
            
            # 转换为与pkl格式兼容的字典
            pred_dict = convert_npz_to_prediction_format(npz_data, token, config)
            # print(pred_dict)
            
            if pred_dict is not None:
                prediction_results.append(pred_dict)
            else:
                failed_files.append(npz_file)
                
        except Exception as e:
            print(f"Warning: 无法加载文件 {npz_file}: {e}")
            failed_files.append(npz_file)
    
    if failed_files:
        print(f"Warning: {len(failed_files)} 个文件加载失败")
    
    if not prediction_results:
        raise ValueError("没有成功加载任何预测结果")

    # print(prediction_results)
    
    print(f"✓ 成功加载 {len(prediction_results)} 个预测结果")
    
    # 验证数据格式
    validate_npz_prediction_format(prediction_results, config)
    
    return prediction_results


def convert_npz_to_prediction_format(npz_data, token, config):
    """
    将npz数据转换为与pkl格式兼容的预测结果字典
    
    支持两种格式：
    1. 标准格式：pts_3d, labels_3d, scores_3d
    2. PivotNet格式：dt_mask, dt_res (包含map, confidence_level, pred_label等)
    
    Args:
        npz_data: 加载的npz数据对象
        token: 样本token
        config: 配置字典
        
    Returns:
        pred_dict: 转换后的预测结果字典，格式与pkl加载器保持一致
    """
    # 获取字段映射配置
    field_mapping = config.get('field_mapping', {})
    
    # 获取npz文件中的键
    npz_keys = list(npz_data.keys())

    # 优先从npz内部提取样本token
    def _extract_sample_token(npz_data, default_token, field_mapping):
        # 1) PivotNet dt_res 内部的 token / sample_idx
        try:
            if 'dt_res' in npz_data:
                dt_res = npz_data['dt_res'].item()
                # print(dt_res)
                for k in ['token', 'sample_idx', 'sample_token']:
                    if k in dt_res:
                        v = _to_python_str(dt_res[k])
                        if v:
                            return v
        except Exception:
            pass

        # 2) 顶层字段，可能是映射的sample_idx字段
        try:
            candidate_keys = [
                field_mapping.get('sample_idx_field', 'sample_idx'),
                'sample_token',
                'token'
            ]
            for k in candidate_keys:
                if k in npz_data:
                    v_raw = npz_data[k]
                    v = _to_python_str(v_raw)
                    if v:
                        return v
        except Exception:
            pass

        return default_token

    token = _extract_sample_token(npz_data, token, field_mapping)
    
    # 检查是否为PivotNet格式
    if 'dt_res' in npz_keys and 'dt_mask' in npz_keys:
        return _convert_pivotnet_format(npz_data, token, config)
    else:
        return _convert_standard_format(npz_data, token, config)


def _convert_pivotnet_format(npz_data, token, config):
    """
    转换PivotNet格式的npz数据
    
    PivotNet格式包含：
    - dt_mask: (3, 400, 200) 分割掩码
    - dt_res: 包含map, confidence_level, pred_label, token, ego_translation, ego_rotation
    """
    try:
        # 提取dt_res中的信息
        dt_res = npz_data['dt_res'].item()
        # 覆盖token为内部提供的值（若存在）
        inner_token = None
        for k in ['token', 'sample_idx', 'sample_token']:
            if k in dt_res:
                inner_token = _to_python_str(dt_res[k])
                if inner_token:
                    break
        if inner_token:
            token = inner_token
        
        # 提取折线数据 (map字段)
        map_data = dt_res.get('map', [])
        confidence_level = dt_res.get('confidence_level', [])
        pred_label = dt_res.get('pred_label', [])
        
        # 过滤掉None值并提取有效的折线
        valid_polylines = []
        valid_labels = []
        valid_scores = []
        
        for i, polyline in enumerate(map_data):
            if polyline is None or len(polyline) == 0:
                continue
            # 标签映射（来自配置：0=None, 1=divider, 2=ped_crossing, 3=boundary）
            # 评测内部使用: 0=divider, 1=ped_crossing, 2=boundary
            mapped_label = None
            if i < len(pred_label):
                raw_label = int(pred_label[i])
                if raw_label == 1:
                    mapped_label = 0
                elif raw_label == 2:
                    mapped_label = 1
                elif raw_label == 3:
                    mapped_label = 2
                else:
                    # raw_label==0 或其它非法，跳过（None 类别不参与评估）
                    mapped_label = None
            # 若缺失标签，视为无效，跳过
            if mapped_label is None:
                continue

            valid_polylines.append(polyline)
            valid_labels.append(mapped_label)

            # 使用confidence_level作为分数
            if i < len(confidence_level):
                score = confidence_level[i]
                # -1 表示无效/未检测，置为 0.0 以便后续过滤
                if score == -1:
                    score = 0.0
                valid_scores.append(float(score))
            else:
                # 未提供分数则视为未检测
                valid_scores.append(0.0)
        
        if not valid_polylines:
            print(f"Warning: 没有找到有效的折线数据")
            return None
        
        # 清洗并末点重复填充（同时同步过滤标签与分数）
        padded_pts_3d, kept_idx = _clean_and_pad_polylines(valid_polylines, token=token, debug=bool(config.get('debug_npz_loader', False)), config=config)
        if padded_pts_3d.shape[0] == 0:
            print(f"Warning: 清洗后无有效折线数据")
            return None

        labels_3d = np.array([valid_labels[i] for i in kept_idx], dtype=np.int64)
        scores_3d = np.array([valid_scores[i] for i in kept_idx], dtype=np.float32)

        # 坐标系处理：优先判断是否为BEV像素坐标；否则执行全局->自车转换
        if bool(config.get('debug_npz_loader', False)):
            flat_pre = padded_pts_3d.reshape(-1, 2)
            if flat_pre.size:
                print(f"[npz_loader][debug] token={token} pre-transform range: x[{flat_pre[:,0].min():.3f},{flat_pre[:,0].max():.3f}], y[{flat_pre[:,1].min():.3f},{flat_pre[:,1].max():.3f}]")

        already_in_ego = False
        try:
            # 从 dt_mask 估计 BEV 尺寸
            bev_W, bev_H = None, None
            if 'dt_mask' in npz_data:
                m = npz_data['dt_mask']
                if hasattr(m, 'shape') and len(m.shape) == 3:
                    # 常见为 (C, W, H) 或 (C, H, W)，取较大者为宽
                    s1, s2 = int(m.shape[1]), int(m.shape[2])
                    bev_W, bev_H = (max(s1, s2), min(s1, s2))
            # 允许从配置覆盖 BEV 尺寸
            bev_cfg = config.get('bev', {}) if isinstance(config, dict) else {}
            if bev_W is None or bev_H is None:
                size = bev_cfg.get('size')
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    bev_W, bev_H = int(size[0]), int(size[1])
            if bev_W is None or bev_H is None:
                bev_W, bev_H = 400, 200

            if _detect_bev_pixel_coords(padded_pts_3d, bev_W, bev_H):
                # print("============================success===========================")
                radius_x_m = float(bev_cfg.get('radius_x_m', 50.0))
                radius_y_m = bev_cfg.get('radius_y_m', None)
                # print(padded_pts_3d)
                
                padded_pts_3d = _bev_pixels_to_ego_meters(padded_pts_3d, bev_W, bev_H, radius_x_m=radius_x_m, radius_y_m=radius_y_m)
                
                # print(padded_pts_3d)
                already_in_ego = True
                # print("===============================end==========================")
        except Exception:
            pass

        if not already_in_ego:
            # 将全局坐标转换为自车坐标
            ego_t = dt_res.get('ego_translation', None)
            ego_q = dt_res.get('ego_rotation', None)
            force_transform = bool(config.get('force_transform_to_ego', True))
            if ego_t is not None and ego_q is not None and force_transform:
                try:
                    # print("success!!!!")
                    t_np = _to_numpy(ego_t)
                    q_np = _to_numpy(ego_q)
                    # print(t_np, q_np)
                    # print(padded_pts_3d)
                    if t_np is not None and q_np is not None:
                        padded_pts_3d = _transform_global_to_ego_batch(padded_pts_3d, t_np, q_np)
                    # print("================================transform_global_to_ego_batch================================================")
                    # print(padded_pts_3d)
                    # print("================================transform_end================================================")
                except Exception:
                    # 回退到启发式转换
                    padded_pts_3d = _maybe_transform_to_ego(padded_pts_3d, ego_t, ego_q, config)
            else:
                # 无ego信息或未强制，则按启发式规则决定
                padded_pts_3d = _maybe_transform_to_ego(padded_pts_3d, ego_t, ego_q, config)

        if bool(config.get('debug_npz_loader', False)):
            flat_post = padded_pts_3d.reshape(-1, 2)
            if flat_post.size:
                print(f"[npz_loader][debug] token={token} post-transform range: x[{flat_post[:,0].min():.3f},{flat_post[:,0].max():.3f}], y[{flat_post[:,1].min():.3f},{flat_post[:,1].max():.3f}]")

        # 自动朝向调整
        padded_pts_3d = _auto_adjust_orientation(padded_pts_3d, config, token=token)

        # 转换为torch.Tensor
        pts_3d_tensor = torch.from_numpy(padded_pts_3d.astype(np.float32))
        labels_3d_tensor = torch.from_numpy(labels_3d.astype(np.int64))
        scores_3d_tensor = torch.from_numpy(scores_3d.astype(np.float32))
        
        # 构建预测结果字典
        pred_dict = {
            'pts_3d': pts_3d_tensor,
            'labels_3d': labels_3d_tensor,
            'scores_3d': scores_3d_tensor,
            'sample_idx': token
        }
        
        # 添加ego信息
        if 'ego_translation' in dt_res:
            pred_dict['ego_translation'] = dt_res['ego_translation']
        if 'ego_rotation' in dt_res:
            pred_dict['ego_rotation'] = dt_res['ego_rotation']

        # print(pred_dict)
        
        return pred_dict
        
    except Exception as e:
        print(f"Warning: 转换PivotNet格式失败: {e}")
        return None


def _convert_standard_format(npz_data, token, config):
    """
    转换标准格式的npz数据 (pts_3d, labels_3d, scores_3d)
    """
    # 获取字段映射配置
    field_mapping = config.get('field_mapping', {})
    
    
    # 获取npz文件中的键
    npz_keys = list(npz_data.keys())
    
    # 根据配置映射字段
    pts_3d_key = field_mapping.get('polylines_field', 'pts_3d')
    labels_3d_key = field_mapping.get('types_field', 'labels_3d')
    scores_3d_key = field_mapping.get('scores_field', 'scores_3d')
    
    # 检查必需字段是否存在
    if pts_3d_key not in npz_keys:
        print(f"Warning: npz文件中缺少字段 {pts_3d_key}")
        return None
    
    if labels_3d_key not in npz_keys:
        print(f"Warning: npz文件中缺少字段 {labels_3d_key}")
        return None
    
    # 提取数据
    pts_3d = npz_data[pts_3d_key]
    labels_3d = npz_data[labels_3d_key]
    
    # 处理分数（如果不存在则使用默认值）
    if scores_3d_key in npz_keys:
        scores_3d = npz_data[scores_3d_key]
    else:
        # 如果没有分数，创建默认分数
        num_polylines = len(pts_3d) if pts_3d.ndim > 1 else 1
        scores_3d = np.ones(num_polylines, dtype=np.float32)
    
    # 统一为 list[np.ndarray] 以便清洗
    polylines_list = []
    if isinstance(pts_3d, np.ndarray) and pts_3d.dtype == object:
        polylines_list = list(pts_3d)
    elif isinstance(pts_3d, np.ndarray) and pts_3d.ndim == 3:
        polylines_list = [pts_3d[i] for i in range(pts_3d.shape[0])]
    elif isinstance(pts_3d, np.ndarray) and pts_3d.ndim == 2:
        polylines_list = [pts_3d]
    else:
        # 其它情况尝试强制转换
        polylines_list = [np.asarray(pts_3d, dtype=np.float32)]

    # 清洗并末点重复填充
    padded_pts_3d, kept_idx = _clean_and_pad_polylines(polylines_list, token=token, debug=bool(config.get('debug_npz_loader', False)), config=config)
    if padded_pts_3d.shape[0] == 0:
        print(f"Warning: 清洗后无有效折线数据: token={token}")
        return None
    
    if isinstance(labels_3d, np.ndarray) and labels_3d.ndim == 0:
        labels_3d = np.array([labels_3d])
    
    if isinstance(scores_3d, np.ndarray) and scores_3d.ndim == 0:
        scores_3d = np.array([scores_3d])

    # 同步过滤标签与分数
    labels_3d = np.asarray(labels_3d, dtype=np.int64)
    scores_3d = np.asarray(scores_3d, dtype=np.float32)
    if labels_3d.shape[0] != len(polylines_list):
        # 若标签数量与原始折线数量不一致，尝试截断/扩展（用最后一个）到相同长度
        if labels_3d.shape[0] == 1 and len(polylines_list) > 1:
            labels_3d = np.repeat(labels_3d, len(polylines_list), axis=0)
        else:
            labels_3d = labels_3d[:len(polylines_list)]
    if scores_3d.shape[0] != len(polylines_list):
        if scores_3d.shape[0] == 1 and len(polylines_list) > 1:
            scores_3d = np.repeat(scores_3d, len(polylines_list), axis=0)
        else:
            scores_3d = scores_3d[:len(polylines_list)]

    labels_3d = labels_3d[kept_idx]
    scores_3d = scores_3d[kept_idx]
    
    # 坐标系处理：优先判断BEV像素坐标；否则全局->自车
    already_in_ego = False
    try:
        bev_cfg = config.get('bev', {}) if isinstance(config, dict) else {}
        size = bev_cfg.get('size', (400, 200))
        bev_W, bev_H = int(size[0]), int(size[1])
        if _detect_bev_pixel_coords(padded_pts_3d, bev_W, bev_H):
            radius_x_m = float(bev_cfg.get('radius_x_m', 50.0))
            radius_y_m = bev_cfg.get('radius_y_m', None)
            padded_pts_3d = _bev_pixels_to_ego_meters(padded_pts_3d, bev_W, bev_H, radius_x_m=radius_x_m, radius_y_m=radius_y_m)
            already_in_ego = True
    except Exception:
        pass

    if not already_in_ego:
        # 从全局转换到自车坐标（若npz提供ego信息字段）
        ego_t_key = field_mapping.get('ego_translation_field', 'ego_translation')
        ego_q_key = field_mapping.get('ego_rotation_field', 'ego_rotation')
        ego_t = npz_data.get(ego_t_key, None)
        ego_q = npz_data.get(ego_q_key, None)
        force_transform = bool(config.get('force_transform_to_ego', True))
        if ego_t is not None and ego_q is not None and force_transform:
            try:
                t_np = _to_numpy(ego_t)
                q_np = _to_numpy(ego_q)
                if t_np is not None and q_np is not None:
                    padded_pts_3d = _transform_global_to_ego_batch(padded_pts_3d, t_np, q_np)
            except Exception:
                # 回退到启发式转换
                padded_pts_3d = _maybe_transform_to_ego(padded_pts_3d, ego_t, ego_q, config)
        else:
            padded_pts_3d = _maybe_transform_to_ego(padded_pts_3d, ego_t, ego_q, config)

    # 自动朝向调整
    padded_pts_3d = _auto_adjust_orientation(padded_pts_3d, config, token=token)

    # 转换为torch.Tensor以保持与pkl格式的一致性
    pts_3d_tensor = torch.from_numpy(padded_pts_3d.astype(np.float32))
    labels_3d_tensor = torch.from_numpy(labels_3d.astype(np.int64))
    scores_3d_tensor = torch.from_numpy(scores_3d.astype(np.float32))
    
    # 构建预测结果字典，格式与pkl加载器保持一致
    pred_dict = {
        'pts_3d': pts_3d_tensor,
        'labels_3d': labels_3d_tensor,
        'scores_3d': scores_3d_tensor,
        'sample_idx': token
    }
    
    # 添加其他可选字段
    for field in ['ego_translation', 'ego_rotation', 'timestamp']:
        field_key = field_mapping.get(f'{field}_field', field)
        if field_key in npz_keys:
            pred_dict[field] = npz_data[field_key]
    
    return pred_dict


def validate_npz_prediction_format(prediction_results, config):
    """
    验证npz预测结果格式是否符合配置要求
    
    Args:
        prediction_results: 预测结果列表
        config: 配置字典
    """
    if not isinstance(prediction_results, list):
        raise ValueError("预测结果必须是列表格式")
    
    if len(prediction_results) == 0:
        raise ValueError("预测结果列表为空")
    
    # 获取字段映射配置
    field_mapping = config.get('field_mapping', {})
    required_fields = field_mapping.get('required_fields', [
        'pts_3d', 'labels_3d', 'scores_3d', 'sample_idx'
    ])
    
    # 检查第一个样本的字段
    sample = prediction_results[0]
    if not isinstance(sample, dict):
        raise ValueError("每个预测结果必须是字典格式")
    
    missing_fields = []
    for field in required_fields:
        if field not in sample:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"预测结果缺少必需字段: {missing_fields}")
    
    # 验证数据形状
    pts_3d = sample['pts_3d']
    labels_3d = sample['labels_3d']
    scores_3d = sample['scores_3d']
    
    if pts_3d.ndim != 3 or pts_3d.shape[2] != 2:
        raise ValueError(f"pts_3d形状错误，期望(N, P, 2)，实际{pts_3d.shape}")
    
    if labels_3d.ndim != 1:
        raise ValueError(f"labels_3d形状错误，期望(N,)，实际{labels_3d.shape}")
    
    if scores_3d.ndim != 1:
        raise ValueError(f"scores_3d形状错误，期望(N,)，实际{scores_3d.shape}")
    
    if len(labels_3d) != len(pts_3d):
        raise ValueError(f"折线数量与标签数量不匹配: {len(pts_3d)} vs {len(labels_3d)}")
    
    if len(scores_3d) != len(pts_3d):
        raise ValueError(f"折线数量与分数数量不匹配: {len(pts_3d)} vs {len(scores_3d)}")
    
    print(f"✓ npz数据格式验证通过")


def parse_npz_prediction_for_stability(prediction_results, config, interval=2, nusc=None, 
                                     pred_rotate_deg=0.0, pred_swap_xy=False, 
                                     pred_flip_x=False, pred_flip_y=False):
    """
    将npz预测结果解析为稳定性评估格式
    复用pkl加载器中的解析逻辑
    
    Args:
        prediction_results: npz预测结果列表
        config: 配置字典
        interval: 帧间隔
        nusc: NuScenes实例，用于获取ego pose信息
        pred_rotate_deg: 预测坐标旋转角度
        pred_swap_xy: 是否交换x/y坐标
        pred_flip_x: 是否翻转x轴
        pred_flip_y: 是否翻转y轴
        
    Returns:
        cur_det_annos: 当前帧检测注释
        pre_det_annos: 前一帧检测注释
        cur_gt_annos: 当前帧GT注释
        pre_gt_annos: 前一帧GT注释
    """
    # 导入pkl加载器中的解析函数
    from .pkl_loader import parse_maptr_data
    
    # 使用pkl加载器的解析逻辑，因为数据格式已经统一
    cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos = parse_maptr_data(
        nusc, prediction_results, interval,
        pred_rotate_deg=pred_rotate_deg,
        pred_swap_xy=pred_swap_xy,
        pred_flip_x=pred_flip_x,
        pred_flip_y=pred_flip_y
    )
    # 标记：在NPZ流程下，稳定性评估阶段采用“历史帧投影到当前帧 + IoU 匹配”的GT实例匹配策略
    try:
        for gt in cur_gt_annos:
            gt['_npz_project_iou_matching'] = True
        for gt in pre_gt_annos:
            gt['_npz_project_iou_matching'] = True
    except Exception:
        pass
    return cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos


def get_npz_file_info(npz_dir):
    """
    获取npz目录的基本信息
    
    Args:
        npz_dir: npz文件目录
        
    Returns:
        info_dict: 包含文件信息的字典
    """
    if not os.path.exists(npz_dir):
        return {"error": f"目录不存在: {npz_dir}"}
    
    npz_pattern = os.path.join(npz_dir, "*.npz")
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        return {"error": f"目录中未找到npz文件: {npz_dir}"}
    
    # 分析第一个文件的内容
    sample_file = npz_files[0]
    try:
        npz_data = np.load(sample_file, allow_pickle=True)
        keys = list(npz_data.keys())
        
        info = {
            "total_files": len(npz_files),
            "sample_file": sample_file,
            "sample_keys": keys,
            "sample_token": Path(sample_file).stem
        }
        
        # 分析数据形状
        for key in keys:
            data = npz_data[key]
            info[f"{key}_shape"] = data.shape if hasattr(data, 'shape') else str(type(data))
            info[f"{key}_dtype"] = str(data.dtype) if hasattr(data, 'dtype') else str(type(data))
        
        return info
        
    except Exception as e:
        return {"error": f"无法分析文件 {sample_file}: {e}"}
