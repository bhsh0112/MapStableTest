"""
NuScenes数据解析模块

包含从NuScenes数据集解析地图元素和MapTR预测结果的功能。
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from shapely.geometry import LineString, box, Polygon
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

from ..geometry import _apply_pred_local_adjust


# MapTR类别定义
MAPTR_CATEGORIES = {
    'ped_crossing': 'ped_crossing',
    'divider': 'divider',
    'boundary': 'boundary'
}


def parse_maptr_data(nusc, pred_infos, interval=1,
                    pred_rotate_deg: float = 0.0,
                    pred_swap_xy: bool = False,
                    pred_flip_x: bool = False,
                    pred_flip_y: bool = False):
    """
    解析MapTR的预测数据和GT数据
    
    Args:
        nusc: NuScenes数据集对象
        pred_infos: 预测信息列表
        interval: 帧间隔
        pred_rotate_deg: 预测结果旋转角度
        pred_swap_xy: 是否交换xy坐标
        pred_flip_x: 是否翻转x轴
        pred_flip_y: 是否翻转y轴
        
    Returns:
        cur_det_annos: 当前帧检测注释
        pre_det_annos: 前一帧检测注释
        cur_gt_annos: 当前帧GT注释
        pre_gt_annos: 前一帧GT注释
    """
    scenes_collector = defaultdict(list)
    for pred_info in pred_infos:
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
        
        # 组织当前帧和前一帧数据
        if len(scene_gt_infos) > interval:
            for i in range(interval, len(scene_gt_infos)):
                cur_det_annos.append(scene_pred_annos[i])
                pre_det_annos.append(scene_pred_annos[i - interval])
                cur_gt_annos.append(scene_gt_infos[i])
                pre_gt_annos.append(scene_gt_infos[i - interval])
    
    return cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos
