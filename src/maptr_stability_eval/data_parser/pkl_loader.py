"""
PKL文件加载和数据解析模块

包含从pkl文件加载预测结果和转换为稳定性评估格式的功能。
与StableMap/tools/test_stable.py中的parse_maptr_data函数完全一致。
"""

import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from shapely.geometry import LineString, box, Polygon
from shapely import affinity, ops
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap

from ..geometry import _apply_pred_local_adjust


def load_prediction_results(pkl_path, config):
    """
    从pkl文件加载预测结果
    
    Args:
        pkl_path: pkl文件路径
        config: 配置字典，包含字段映射信息
        
    Returns:
        prediction_results: 预测结果列表
    """
    print(f"加载预测结果文件: {pkl_path}")
    
    try:
        with open(pkl_path, 'rb') as f:
            prediction_results = pickle.load(f)
    except Exception as e:
        raise ValueError(f"无法加载pkl文件 {pkl_path}: {e}")
    
    print(f"✓ 成功加载 {len(prediction_results)} 个预测结果")
    
    # 验证数据格式
    validate_prediction_format(prediction_results, config)
    
    return prediction_results


def validate_prediction_format(prediction_results, config):
    """
    验证预测结果格式是否符合配置要求
    
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
    required_fields = field_mapping.get('required_fields', [])
    
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
    
    print(f"✓ 数据格式验证通过")


def parse_prediction_for_stability(prediction_results, config, interval=2, nusc=None, 
                                 pred_rotate_deg=0.0, pred_swap_xy=False, 
                                 pred_flip_x=False, pred_flip_y=False):
    """
    将预测结果解析为稳定性评估格式
    与StableMap/tools/test_stable.py中的parse_maptr_data函数完全一致
    
    Args:
        prediction_results: 预测结果列表
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
    print(f"解析预测数据，帧间隔: {interval}")
    
    if nusc is None:
        raise ValueError("NuScenes实例不能为空，需要用于获取真实的地图GT数据")
    
    # 按场景分组 - 与test_stable.py完全一致
    scenes_collector = defaultdict(list)
    for pred_info in prediction_results:
        sample_token = pred_info['sample_idx']
        try:
            gt_sample = nusc.get('sample', sample_token)
            scene_token = gt_sample['scene_token']
            scenes_collector[scene_token].append([gt_sample, pred_info])
        except KeyError:
            print(f"Warning: Sample token {sample_token} not found in NuScenes dataset. Skipping.")
            continue
    
    cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos = [], [], [], []
    
    for scene_token, info_list in tqdm(list(scenes_collector.items())):
        info_list = sorted(info_list, key=lambda x: x[0]['timestamp'])
        scene_gt_samples, scene_pred_infos = list(zip(*info_list))
        
        # 获取地图数据 - 与test_stable.py完全一致
        from nuscenes.map_expansion.map_api import NuScenesMap
        log = nusc.get('scene', scene_token)['log_token']
        map_name = nusc.get('log', log)['location']
        nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)
        
        scene_gt_infos = []
        for sample, pred_info in zip(scene_gt_samples, scene_pred_infos):
            # 获取ego pose信息 - 与test_stable.py完全一致
            sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            ego_translation = np.array(pose_record['translation'])
            ego_rotation = Quaternion(pose_record['rotation'])
            
            # 获取地图元素 - 与test_stable.py完全一致
            polylines = []
            instance_ids = []  # 存储 (layer, token) 组合作为实例ID
            types = []
            radius = 50  # 搜索半径50米
            
            # 坐标转换函数 - 与test_stable.py完全一致
            def transform_points(points, ego_translation, ego_rotation):
                points = points - ego_translation[:2]
                rot = ego_rotation.inverse
                # 确保points是二维数组
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                points = np.dot(rot.rotation_matrix[:2, :2], points.T).T
                return points
            
            def extract_polygon_boundaries(polygon):
                boundaries = []
                # 外边界
                if not polygon.exterior.is_empty:
                    boundaries.append(polygon.exterior)
                # 内边界（洞）
                for interior in polygon.interiors:
                    if not interior.is_empty:
                        boundaries.append(interior)
                return boundaries
            
            # 1. 提取车道分隔线 (divider) - 与test_stable.py完全一致
            for layer in ['lane_divider', 'road_divider']:
                records = nusc_map.get_records_in_radius(
                    ego_translation[0], ego_translation[1], radius, [layer]
                )
                if layer in records:
                    for token in records[layer]:
                        record = nusc_map.get(layer, token)
                        line_points = []
                        if 'node_tokens' in record:
                            for node_token in record['node_tokens']:
                                node_record = nusc_map.get('node', node_token)
                                if 'x' in node_record and 'y' in node_record:
                                    line_points.append([node_record['x'], node_record['y']])
                            line_points = np.array(line_points)
                            line_points = transform_points(line_points, ego_translation, ego_rotation)
                            
                            # 创建LineString对象进行几何操作
                            line = LineString(line_points)
                            
                            # 裁剪到局部坐标系（50x50米）
                            max_x, max_y = 25, 25
                            local_patch = box(-max_x, -max_y, max_x, max_y)
                            clipped_line = line.intersection(local_patch)
                            
                            if clipped_line.is_empty:
                                continue
                                
                            # 处理可能的多段线
                            if clipped_line.geom_type == 'MultiLineString':
                                for part in clipped_line.geoms:
                                    points = np.array(part.coords)[:, :2]
                                    polylines.append(points)
                                    instance_ids.append(f"{layer}_{token}")
                                    types.append('divider')
                            else:
                                points = np.array(clipped_line.coords)[:, :2]
                                polylines.append(points)
                                instance_ids.append(f"{layer}_{token}")
                                types.append('divider')
            
            # 2. 提取人行横道 (ped_crossing) - 与test_stable.py完全一致
            layer = 'ped_crossing'
            records = nusc_map.get_records_in_radius(
                ego_translation[0], ego_translation[1], radius, [layer]
            )
            if layer in records:
                for token in records[layer]:
                    record = nusc_map.get(layer, token)
                    if 'polygon_token' in record:
                        polygon = nusc_map.extract_polygon(record['polygon_token'])
                        
                        # 跳过无效多边形
                        if not polygon.is_valid or polygon.is_empty:
                            continue
                        
                        # 转换到自车坐标系
                        poly_points = np.array(polygon.exterior.coords)[:, :2]
                        poly_points = transform_points(poly_points, ego_translation, ego_rotation)
                        
                        # 创建多边形对象
                        polygon = Polygon(poly_points)
                        
                        # 裁剪到局部坐标系
                        max_x, max_y = 25, 25
                        local_patch = box(-max_x, -max_y, max_x, max_y)
                        clipped_poly = polygon.intersection(local_patch)
                        
                        if clipped_poly.is_empty:
                            continue
                        
                        # 处理不同类型的裁剪结果
                        if clipped_poly.geom_type == 'Polygon':
                            boundaries = extract_polygon_boundaries(clipped_poly)
                        elif clipped_poly.geom_type == 'MultiPolygon':
                            boundaries = []
                            for poly in clipped_poly.geoms:
                                boundaries.extend(extract_polygon_boundaries(poly))
                        
                        # 处理每个边界
                        for boundary in boundaries:
                            # 调整方向（外边界逆时针→顺时针，内边界顺时针→逆时针）
                            if boundary.is_ring and boundary.is_ccw:  # 外边界
                                boundary = LineString(list(boundary.coords)[::-1])
                            elif not boundary.is_ccw:  # 内边界
                                boundary = LineString(list(boundary.coords)[::-1])
                            
                            # 添加边界线
                            points = np.array(boundary.coords)[:, :2]
                            polylines.append(points)
                            instance_ids.append(f"ped_{token}")
                            types.append('ped_crossing')
            
            # 3. 提取道路边界 (boundary) - 与test_stable.py完全一致
            boundary_layers = ['road_segment', 'lane']
            for layer in boundary_layers:
                records = nusc_map.get_records_in_radius(
                    ego_translation[0], ego_translation[1], radius, [layer]
                )
                if layer not in records:
                    continue
                    
                for token in records[layer]:
                    record = nusc_map.get(layer, token)
                    polygon = nusc_map.extract_polygon(record['polygon_token'])
                    
                    # 提取所有边界线（外边界+内边界）
                    boundaries = extract_polygon_boundaries(polygon)
                    
                    for boundary in boundaries:
                         # 获取全局坐标点
                        points = np.array(boundary.coords)[:, :2]
                        
                        # 转换到自车坐标系
                        points_ego = transform_points(points, ego_translation, ego_rotation)
                        
                        # 创建LineString对象（在自车坐标系中）
                        line = LineString(points_ego)
                        
                        # 裁剪到局部坐标系（50x50米）
                        max_x, max_y = 25, 25
                        local_patch = box(-max_x, -max_y, max_x, max_y)
                        clipped_boundary = line.intersection(local_patch)
                        
                        if clipped_boundary.is_empty:
                            continue
                            
                        # 处理裁剪结果（可能是MultiLineString）
                        if clipped_boundary.geom_type == 'MultiLineString':
                            for part in clipped_boundary.geoms:
                                if part.is_empty:
                                    continue
                                part_points = np.array(part.coords)[:, :2]
                                polylines.append(part_points)
                                instance_ids.append(f"boundary_{layer}_{token}")
                                types.append('boundary')
                        else:
                            part_points = np.array(clipped_boundary.coords)[:, :2]
                            polylines.append(part_points)
                            instance_ids.append(f"boundary_{layer}_{token}")
                            types.append('boundary')
            
            # 构建GT注释 - 与test_stable.py完全一致
            scene_gt_infos.append({
                'polylines': polylines,
                'instance_ids': np.array(instance_ids),
                'types': np.array(types),
                'token': sample['token'],
                'timestamp': sample['timestamp'],
                'ego_translation': ego_translation,
                'ego_rotation': ego_rotation
            })
        
        # 构建预测注释 - 与test_stable.py完全一致
        scene_pred_annos = []
        for pred_info in scene_pred_infos:
            # 解析预测结果
            labels = pred_info['labels_3d'].cpu().numpy()
            pts_3d = pred_info['pts_3d'].cpu().numpy()
            
            # 映射类别ID到名称
            class_mapping = {
                0: 'divider',
                1: 'ped_crossing',
                2: 'boundary'
            }
            pred_types = [class_mapping.get(label, 'unknown') for label in labels]

            if 'scores_3d' in pred_info:
                scores = pred_info['scores_3d'].cpu().numpy().tolist()
            else:
                scores = [1.0] * len(pts_3d)
            
            # 构建预测注释字典 - 与test_stable.py完全一致
            scene_pred_annos.append({
                'polylines': [_apply_pred_local_adjust(pts, rotate_deg=pred_rotate_deg,
                                                       swap_xy=pred_swap_xy, flip_x=pred_flip_x, flip_y=pred_flip_y)
                              for pts in pts_3d],
                'types': pred_types,
                'scores': scores,  
                'sample_idx': pred_info['sample_idx']
            })

        # 组织当前帧和前一帧数据 - 与test_stable.py完全一致
        if len(scene_gt_infos) > interval:
            for i in range(interval, len(scene_gt_infos)):
                cur_det_annos.append(scene_pred_annos[i])
                pre_det_annos.append(scene_pred_annos[i - interval])
                cur_gt_annos.append(scene_gt_infos[i])
                pre_gt_annos.append(scene_gt_infos[i - interval])
    
    print(f"✓ 数据解析完成，生成 {len(cur_det_annos)} 对帧数据")
    return cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos

def parse_maptr_data(nusc, pred_infos, interval=1,
                    pred_rotate_deg: float = 0.0,
                    pred_swap_xy: bool = False,
                    pred_flip_x: bool = False,
                    pred_flip_y: bool = False):
    """
    解析MapTR的预测数据和GT数据
    """
    scenes_collector = defaultdict(list)
    for pred_info in pred_infos:
        sample_token = pred_info['sample_idx']
        # print("=======================")
        # print(sample_token)
        try:
            gt_sample = nusc.get('sample', sample_token)
            scene_token = gt_sample['scene_token']
            scenes_collector[scene_token].append([gt_sample, pred_info])
        except KeyError:
            print(f"Warning: Sample token {sample_token} not found in NuScenes dataset. Skipping.")
            continue
    
    cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos = [], [], [], []
    
    for scene_token, info_list in tqdm(list(scenes_collector.items())):
        info_list = sorted(info_list, key=lambda x: x[0]['timestamp'])
        scene_gt_samples, scene_pred_infos = list(zip(*info_list))
        
        # 获取地图数据
        log = nusc.get('scene', scene_token)['log_token']
        map_name = nusc.get('log', log)['location']
        nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)
        
        scene_gt_infos = []
        for sample, pred_info in zip(scene_gt_samples, scene_pred_infos):
            # 获取ego pose信息
            sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            ego_translation = np.array(pose_record['translation'])
            ego_rotation = Quaternion(pose_record['rotation'])
            
            # 获取地图元素
            polylines = []
            instance_ids = []  # 存储 (layer, token) 组合作为实例ID
            types = []
            radius = 50  # 搜索半径50米
            
            # 坐标转换函数
            def transform_points(points, ego_translation, ego_rotation):
                points = points - ego_translation[:2]
                rot = ego_rotation.inverse
                # 确保points是二维数组
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                points = np.dot(rot.rotation_matrix[:2, :2], points.T).T
                return points
            
            def extract_polygon_boundaries(polygon):
                boundaries = []
                # 外边界
                if not polygon.exterior.is_empty:
                    boundaries.append(polygon.exterior)
                # 内边界（洞）
                for interior in polygon.interiors:
                    if not interior.is_empty:
                        boundaries.append(interior)
                return boundaries
            
            # 1. 提取车道分隔线 (divider)
            for layer in ['lane_divider', 'road_divider']:
                records = nusc_map.get_records_in_radius(
                    ego_translation[0], ego_translation[1], radius, [layer]
                )
                if layer in records:
                    # print("success!!!!!!")
                    for token in records[layer]:
                        record = nusc_map.get(layer, token)
                        line_points = []
                        if 'node_tokens' in record:
                            for node_token in record['node_tokens']:
                                node_record = nusc_map.get('node', node_token)
                                if 'x' in node_record and 'y' in node_record:
                                    line_points.append([node_record['x'], node_record['y']])
                            line_points = np.array(line_points)
                            line_points = transform_points(line_points, ego_translation, ego_rotation)
                            
                            # 创建LineString对象进行几何操作
                            line = LineString(line_points)
                            
                            # 裁剪到局部坐标系（50x50米）
                            max_x, max_y = 25, 25
                            local_patch = box(-max_x, -max_y, max_x, max_y)
                            clipped_line = line.intersection(local_patch)
                            
                            if clipped_line.is_empty:
                                continue
                                
                            # 处理可能的多段线
                            if clipped_line.geom_type == 'MultiLineString':
                                for part in clipped_line.geoms:
                                    points = np.array(part.coords)[:, :2]
                                    polylines.append(points)
                                    instance_ids.append(f"{layer}_{token}")
                                    types.append('divider')
                            else:
                                points = np.array(clipped_line.coords)[:, :2]
                                polylines.append(points)
                                instance_ids.append(f"{layer}_{token}")
                                types.append('divider')
            
            # 2. 提取人行横道 (ped_crossing)
            layer = 'ped_crossing'
            records = nusc_map.get_records_in_radius(
                ego_translation[0], ego_translation[1], radius, [layer]
            )
            if layer in records:
                for token in records[layer]:
                    record = nusc_map.get(layer, token)
                    if 'polygon_token' in record:
                        polygon = nusc_map.extract_polygon(record['polygon_token'])
                        
                        # 跳过无效多边形
                        if not polygon.is_valid or polygon.is_empty:
                            continue
                        
                        # 转换到自车坐标系
                        poly_points = np.array(polygon.exterior.coords)[:, :2]
                        poly_points = transform_points(poly_points, ego_translation, ego_rotation)
                        
                        # 创建多边形对象
                        polygon = Polygon(poly_points)
                        
                        # 裁剪到局部坐标系
                        max_x, max_y = 25, 25
                        local_patch = box(-max_x, -max_y, max_x, max_y)
                        clipped_poly = polygon.intersection(local_patch)
                        
                        if clipped_poly.is_empty:
                            continue
                        
                        # 处理不同类型的裁剪结果
                        if clipped_poly.geom_type == 'Polygon':
                            boundaries = extract_polygon_boundaries(clipped_poly)
                        elif clipped_poly.geom_type == 'MultiPolygon':
                            boundaries = []
                            for poly in clipped_poly.geoms:
                                boundaries.extend(extract_polygon_boundaries(poly))
                        
                        # 处理每个边界
                        for boundary in boundaries:
                            # 调整方向（外边界逆时针→顺时针，内边界顺时针→逆时针）
                            if boundary.is_ring and boundary.is_ccw:  # 外边界
                                boundary = LineString(list(boundary.coords)[::-1])
                            elif not boundary.is_ccw:  # 内边界
                                boundary = LineString(list(boundary.coords)[::-1])
                            
                            # 添加边界线
                            points = np.array(boundary.coords)[:, :2]
                            polylines.append(points)
                            instance_ids.append(f"ped_{token}")
                            types.append('ped_crossing')
            
            # 3. 提取道路边界 (boundary)
            boundary_layers = ['road_segment', 'lane']
            for layer in boundary_layers:
                records = nusc_map.get_records_in_radius(
                    ego_translation[0], ego_translation[1], radius, [layer]
                )
                if layer not in records:
                    continue
                    
                for token in records[layer]:
                    record = nusc_map.get(layer, token)
                    polygon = nusc_map.extract_polygon(record['polygon_token'])
                    
                    # 提取所有边界线（外边界+内边界）
                    boundaries = extract_polygon_boundaries(polygon)

                    # print("success!!!!!!!!!!!!")
                    # print(boundaries)
                    
                    for boundary in boundaries:
                         # 获取全局坐标点
                        points = np.array(boundary.coords)[:, :2]
                        
                        # 转换到自车坐标系
                        points_ego = transform_points(points, ego_translation, ego_rotation)
                        
                        # 创建LineString对象（在自车坐标系中）
                        line = LineString(points_ego)
                        
                        # 裁剪到局部坐标系（50x50米）
                        max_x, max_y = 25, 25
                        local_patch = box(-max_x, -max_y, max_x, max_y)
                        clipped_boundary = line.intersection(local_patch)
                        
                        if clipped_boundary.is_empty:
                            continue
                            
                        # 处理裁剪结果（可能是MultiLineString）
                        if clipped_boundary.geom_type == 'MultiLineString':
                            for part in clipped_boundary.geoms:
                                if part.is_empty:
                                    continue
                                part_points = np.array(part.coords)[:, :2]
                                polylines.append(part_points)
                                instance_ids.append(f"boundary_{layer}_{token}")
                                types.append('boundary')
                        else:
                            part_points = np.array(clipped_boundary.coords)[:, :2]
                            polylines.append(part_points)
                            instance_ids.append(f"boundary_{layer}_{token}")
                            types.append('boundary')
            
            # print(types)
            # 构建GT注释
            scene_gt_infos.append({
                'polylines': polylines,
                'instance_ids': np.array(instance_ids),
                'types': np.array(types),
                'token': sample['token'],
                'timestamp': sample['timestamp'],
                'ego_translation': ego_translation,
                'ego_rotation': ego_rotation
            })
        
        # 构建预测注释
        scene_pred_annos = []
        for pred_info in scene_pred_infos:
            # 解析预测结果
            labels = pred_info['labels_3d'].cpu().numpy()
            pts_3d = pred_info['pts_3d'].cpu().numpy()
            
            # 映射类别ID到名称
            class_mapping = {
                0: 'divider',
                1: 'ped_crossing',
                2: 'boundary'
            }
            pred_types = [class_mapping.get(label, 'unknown') for label in labels]

            if 'scores_3d' in pred_info:
                scores = pred_info['scores_3d'].cpu().numpy().tolist()
            else:
                scores = [1.0] * len(pts_3d)
            
            # 构建预测注释字典
            scene_pred_annos.append({
                'polylines': [_apply_pred_local_adjust(pts, rotate_deg=pred_rotate_deg,
                                                       swap_xy=pred_swap_xy, flip_x=pred_flip_x, flip_y=pred_flip_y)
                              for pts in pts_3d],
                'types': pred_types,
                'scores': scores,  
                'sample_idx': pred_info['sample_idx']
            })

        # print(scene_pred_annos)
        # print("=============================================")
        # print(scene_gt_infos)
        
        # 组织当前帧和前一帧数据
        if len(scene_gt_infos) > interval:
            for i in range(interval, len(scene_gt_infos)):
                cur_det_annos.append(scene_pred_annos[i])
                pre_det_annos.append(scene_pred_annos[i - interval])
                cur_gt_annos.append(scene_gt_infos[i])
                pre_gt_annos.append(scene_gt_infos[i - interval])
    
    return cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos

