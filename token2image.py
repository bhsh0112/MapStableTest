#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NuScenes Token to Image 转换工具

该脚本用于根据NuScenes数据集的token输出对应的输入图像，包括：
- 多相机图像（6个视角）
- 激光雷达点云数据
- 自车位置和姿态信息
- 地图信息

作者: AI Assistant
日期: 2024
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

# NuScenes相关导入
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion


class NuScenesTokenToImage:
    """
    NuScenes Token到图像转换器
    
    该类提供了从NuScenes数据集的token提取和可视化各种输入数据的功能
    """
    
    def __init__(self, dataroot: str, version: str = 'v1.0-mini', verbose: bool = True):
        """
        初始化NuScenes数据集
        
        Args:
            dataroot: NuScenes数据集根目录路径
            version: 数据集版本 (v1.0-mini, v1.0-trainval, v1.0-test)
            verbose: 是否显示详细信息
        """
        self.dataroot = dataroot
        self.version = version
        self.verbose = verbose
        
        # 初始化NuScenes实例
        try:
            self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
            print(f"✅ 成功加载NuScenes数据集: {version}")
        except Exception as e:
            print(f"❌ 加载NuScenes数据集失败: {e}")
            sys.exit(1)
        
        # 相机配置
        self.camera_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]
        
        # 颜色配置
        self.colors = {
            'CAM_FRONT': 'red',
            'CAM_FRONT_RIGHT': 'orange', 
            'CAM_BACK_RIGHT': 'yellow',
            'CAM_BACK': 'green',
            'CAM_BACK_LEFT': 'blue',
            'CAM_FRONT_LEFT': 'purple'
        }
    
    def get_sample_info(self, sample_token: str) -> Dict:
        """
        获取sample的基本信息
        
        Args:
            sample_token: sample的token
            
        Returns:
            包含sample信息的字典
        """
        try:
            sample = self.nusc.get('sample', sample_token)
            
            # 获取场景信息
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            
            info = {
                'sample_token': sample_token,
                'scene_token': sample['scene_token'],
                'timestamp': sample['timestamp'],
                'scene_name': scene['name'],
                'location': log['location'],
                'weather': log['weather'],
                'camera_data': {},
                'lidar_data': None,
                'ego_pose': None
            }
            
            return info
            
        except Exception as e:
            print(f"❌ 获取sample信息失败: {e}")
            return None
    
    def extract_camera_images(self, sample_token: str, save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        提取指定sample的所有相机图像
        
        Args:
            sample_token: sample的token
            save_dir: 图像保存目录，如果为None则不保存
            
        Returns:
            包含所有相机图像的字典
        """
        try:
            sample = self.nusc.get('sample', sample_token)
            images = {}
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            for camera_name in self.camera_names:
                # 获取相机数据token
                camera_token = sample['data'][camera_name]
                camera_data = self.nusc.get('sample_data', camera_token)
                
                # 构建图像路径
                image_path = os.path.join(self.dataroot, camera_data['filename'])
                
                # 读取图像
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images[camera_name] = image
                    
                    # 保存图像
                    if save_dir:
                        save_path = os.path.join(save_dir, f"{camera_name}_{sample_token[:8]}.jpg")
                        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        print(f"📷 保存相机图像: {save_path}")
                else:
                    print(f"⚠️  图像文件不存在: {image_path}")
            
            return images
            
        except Exception as e:
            print(f"❌ 提取相机图像失败: {e}")
            return {}
    
    def extract_lidar_data(self, sample_token: str, save_dir: Optional[str] = None) -> Optional[np.ndarray]:
        """
        提取激光雷达点云数据
        
        Args:
            sample_token: sample的token
            save_dir: 点云保存目录，如果为None则不保存
            
        Returns:
            点云数据数组 (N, 4) [x, y, z, intensity]
        """
        try:
            sample = self.nusc.get('sample', sample_token)
            
            # 获取激光雷达数据token
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            
            # 构建点云文件路径
            pc_path = os.path.join(self.dataroot, lidar_data['filename'])
            
            if os.path.exists(pc_path):
                # 读取点云数据
                pc = LidarPointCloud.from_file(pc_path)
                points = pc.points.T  # 转换为 (N, 4) 格式
                
                # 保存点云数据
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"lidar_{sample_token[:8]}.npy")
                    np.save(save_path, points)
                    print(f"🔍 保存激光雷达数据: {save_path}")
                
                return points
            else:
                print(f"⚠️  点云文件不存在: {pc_path}")
                return None
                
        except Exception as e:
            print(f"❌ 提取激光雷达数据失败: {e}")
            return None
    
    def get_ego_pose(self, sample_token: str) -> Optional[Dict]:
        """
        获取自车位置和姿态信息
        
        Args:
            sample_token: sample的token
            
        Returns:
            包含自车位姿信息的字典
        """
        try:
            sample = self.nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            # 提取位置和旋转信息
            translation = ego_pose['translation']
            rotation = ego_pose['rotation']
            
            # 将四元数转换为欧拉角
            q = Quaternion(rotation)
            yaw, pitch, roll = q.yaw_pitch_roll
            
            pose_info = {
                'translation': translation,
                'rotation_quat': rotation,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'timestamp': ego_pose['timestamp']
            }
            
            return pose_info
            
        except Exception as e:
            print(f"❌ 获取自车位姿失败: {e}")
            return None
    
    def visualize_camera_images(self, images: Dict[str, np.ndarray], sample_token: str, 
                              save_path: Optional[str] = None) -> None:
        """
        可视化所有相机图像
        
        Args:
            images: 相机图像字典
            sample_token: sample的token
            save_path: 可视化结果保存路径
        """
        if not images:
            print("⚠️  没有可可视化的图像")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (camera_name, image) in enumerate(images.items()):
            if i < len(axes):
                axes[i].imshow(image)
                axes[i].set_title(f'{camera_name}\nToken: {sample_token[:8]}...', 
                                color=self.colors.get(camera_name, 'black'))
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'NuScenes Sample Images - Token: {sample_token}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🖼️  保存可视化图像: {save_path}")
        
        plt.show()
    
    def visualize_lidar_points(self, points: np.ndarray, sample_token: str, 
                             save_path: Optional[str] = None) -> None:
        """
        可视化激光雷达点云（俯视图）
        
        Args:
            points: 点云数据 (N, 4)
            sample_token: sample的token
            save_path: 可视化结果保存路径
        """
        if points is None or len(points) == 0:
            print("⚠️  没有可可视化的点云数据")
            return
        
        # 创建俯视图
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # 绘制点云（使用强度作为颜色）
        scatter = ax.scatter(points[:, 0], points[:, 1], 
                           c=points[:, 3], cmap='viridis', s=0.5, alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'LiDAR Point Cloud - Token: {sample_token[:8]}...')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='Intensity')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🔍 保存点云可视化: {save_path}")
        
        plt.show()
    
    def process_token(self, sample_token: str, output_dir: str = "./output") -> Dict:
        """
        处理指定的token，提取所有相关数据
        
        Args:
            sample_token: sample的token
            output_dir: 输出目录
            
        Returns:
            包含所有提取数据的字典
        """
        print(f"🚀 开始处理token: {sample_token}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取sample基本信息
        sample_info = self.get_sample_info(sample_token)
        if not sample_info:
            return {}
        
        print(f"📍 场景: {sample_info['scene_name']}")
        print(f"🌍 位置: {sample_info['location']}")
        print(f"🌤️  天气: {sample_info['weather']}")
        
        # 提取相机图像
        print("📷 提取相机图像...")
        images = self.extract_camera_images(sample_token, 
                                          os.path.join(output_dir, "camera_images"))
        
        # 提取激光雷达数据
        print("🔍 提取激光雷达数据...")
        lidar_points = self.extract_lidar_data(sample_token, 
                                             os.path.join(output_dir, "lidar_data"))
        
        # 获取自车位姿
        print("🚗 获取自车位姿...")
        ego_pose = self.get_ego_pose(sample_token)
        
        # 可视化相机图像
        if images:
            camera_viz_path = os.path.join(output_dir, f"camera_visualization_{sample_token[:8]}.png")
            self.visualize_camera_images(images, sample_token, camera_viz_path)
        
        # 可视化激光雷达点云
        if lidar_points is not None:
            lidar_viz_path = os.path.join(output_dir, f"lidar_visualization_{sample_token[:8]}.png")
            self.visualize_lidar_points(lidar_points, sample_token, lidar_viz_path)
        
        # 保存元数据
        metadata = {
            'sample_token': sample_token,
            'sample_info': sample_info,
            'ego_pose': ego_pose,
            'num_cameras': len(images),
            'num_lidar_points': len(lidar_points) if lidar_points is not None else 0,
            'camera_names': list(images.keys()),
            'output_directory': output_dir
        }
        
        metadata_path = os.path.join(output_dir, f"metadata_{sample_token[:8]}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📄 保存元数据: {metadata_path}")
        print(f"✅ 处理完成！输出目录: {output_dir}")
        
        return metadata


def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(description='NuScenes Token到图像转换工具')
    parser.add_argument('--token', type=str, required=True, 
                       help='NuScenes sample token')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='NuScenes数据集根目录路径')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                       choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                       help='NuScenes数据集版本')
    parser.add_argument('--output', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--no-visualize', action='store_true',
                       help='不显示可视化图像')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.dataroot):
        print(f"❌ 数据集路径不存在: {args.dataroot}")
        sys.exit(1)
    
    # 创建转换器实例
    converter = NuScenesTokenToImage(
        dataroot=args.dataroot,
        version=args.version,
        verbose=True
    )
    
    # 处理token
    try:
        result = converter.process_token(args.token, args.output)
        
        if result:
            print("\n📊 处理结果摘要:")
            print(f"   Token: {result['sample_token']}")
            print(f"   相机数量: {result['num_cameras']}")
            print(f"   点云数量: {result['num_lidar_points']}")
            print(f"   输出目录: {result['output_directory']}")
        else:
            print("❌ 处理失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 如果直接运行脚本，使用命令行接口
    if len(sys.argv) > 1:
        main()
    else:
        # 交互式使用示例
        print("🔧 NuScenes Token到图像转换工具")
        print("=" * 50)
        
        # 配置参数（请根据实际情况修改）
        DATAROOT = "/path/to/nuscenes"  # 请修改为实际的数据集路径
        VERSION = "v1.0-mini"
        SAMPLE_TOKEN = "0a0d6b8c2e884134a3b48df43d54c36a"  # 示例token
        
        print(f"📁 数据集路径: {DATAROOT}")
        print(f"📋 数据集版本: {VERSION}")
        print(f"🔑 示例Token: {SAMPLE_TOKEN}")
        print("\n💡 使用方法:")
        print("   python token2image.py --token <TOKEN> --dataroot <PATH> --version <VERSION>")
        print("\n📖 或者修改脚本中的配置参数后直接运行")
