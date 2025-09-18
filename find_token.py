import matplotlib
# 使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
import os
from PIL import Image
import io

def save_sample_and_prev_map(nusc, sample_token, save_dir="./output_maps"):
    """
    保存指定sample及其前一帧的地图图像（兼容不同版本的NuScenes API）
    
    参数:
    nusc: NuScenes实例
    sample_token: 目标sample的token
    save_dir: 图像保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取目标sample记录
    target_sample = nusc.get('sample', sample_token)
    
    # 获取前一帧sample token
    prev_sample_token = target_sample['prev']
    
    if not prev_sample_token:
        print("目标sample没有前一帧（可能是场景的第一帧）")
        return None
    
    # 获取目标sample和前一帧sample的ego_pose
    target_sample_data = nusc.get('sample_data', target_sample['data']['LIDAR_TOP'])
    target_ego_pose = nusc.get('ego_pose', target_sample_data['ego_pose_token'])
    
    prev_sample = nusc.get('sample', prev_sample_token)
    prev_sample_data = nusc.get('sample_data', prev_sample['data']['LIDAR_TOP'])
    prev_ego_pose = nusc.get('ego_pose', prev_sample_data['ego_pose_token'])
    
    # 获取场景记录以确定地图名称
    scene_token = target_sample['scene_token']
    scene_record = nusc.get('scene', scene_token)
    log_token = scene_record['log_token']
    log_record = nusc.get('log', log_token)
    map_name = log_record['location']
    
    # 初始化地图
    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 定义地图渲染函数（处理不同版本的API）
    def render_map_patch_compatible(nusc_map, box_coords, layer_names, ax=None):
        """
        兼容不同版本NuScenes API的地图渲染函数
        """
        try:
            # 尝试新版本的API（可能不需要ax参数）
            if ax is None:
                return nusc_map.render_map_patch(box_coords, layer_names, alpha=0.5)
            else:
                # 尝试旧版本的API（需要figsize参数而不是ax参数）
                return nusc_map.render_map_patch(box_coords, layer_names, figsize=10, alpha=0.5)
        except TypeError as e:
            # 如果仍然出错，使用替代方法
            print(f"渲染地图时出现兼容性问题: {e}")
            print("使用替代渲染方法...")
            
            # 尝试直接渲染到图像缓冲区
            try:
                img_buffer = io.BytesIO()
                plt.figure(figsize=(8, 8))
                nusc_map.render_map_patch(box_coords, layer_names, alpha=0.5)
                plt.savefig(img_buffer, format='png', dpi=100)
                plt.close()
                
                # 将图像加载回并添加到子图中
                img = Image.open(img_buffer)
                ax.imshow(img)
                ax.set_title("地图渲染")
                ax.axis('off')
                return ax
            except Exception as e2:
                print(f"替代方法也失败: {e2}")
                # 最后尝试使用explorer类的方法
                try:
                    from nuscenes.nuscenes import NuScenesExplorer
                    explorer = NuScenesExplorer(nusc)
                    explorer.render_egoposes_on_fancy_map(
                        [target_ego_pose['token'], prev_ego_pose['token']], 
                        nusc_map, 
                        axes=ax
                    )
                except Exception as e3:
                    print(f"所有渲染方法都失败: {e3}")
                    ax.text(0.5, 0.5, '地图渲染失败', ha='center', va='center')
                return ax
    
    # 绘制目标帧地图
    ax1.set_title(f"目标帧 (Token: {sample_token[:8]}...)")
    
    # 定义地图区域
    target_box_coords = [
        target_ego_pose['translation'][0] - 40, 
        target_ego_pose['translation'][1] - 40,
        target_ego_pose['translation'][0] + 40, 
        target_ego_pose['translation'][1] + 40
    ]
    
    # 渲染地图
    render_map_patch_compatible(
        nusc_map, 
        target_box_coords, 
        ['drivable_area', 'walkway'], 
        ax=ax1
    )
    
    # 绘制自车位置（目标帧）
    ax1.plot(target_ego_pose['translation'][0], target_ego_pose['translation'][1], 
            'ro', markersize=10, label='Ego Vehicle')
    
    # 绘制自车方向（目标帧）
    # 提取旋转矩阵中的偏航角（简化处理）
    rotation = target_ego_pose['rotation']
    if isinstance(rotation, list) and len(rotation) >= 4:
        # 四元数转换为偏航角
        from pyquaternion import Quaternion
        q = Quaternion(rotation)
        yaw = q.yaw_pitch_roll[0]
    else:
        # 使用简化的偏航角提取
        yaw = 0  # 默认值
    
    dx, dy = 5 * np.cos(yaw), 5 * np.sin(yaw)
    ax1.arrow(target_ego_pose['translation'][0], target_ego_pose['translation'][1],
             dx, dy, head_width=2, head_length=2, fc='r', ec='r')
    
    ax1.legend()
    ax1.grid(False)
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    
    # 绘制前一帧地图
    ax2.set_title(f"前一帧 (Token: {prev_sample_token[:8]}...)")
    
    # 定义地图区域
    prev_box_coords = [
        prev_ego_pose['translation'][0] - 40, 
        prev_ego_pose['translation'][1] - 40,
        prev_ego_pose['translation'][0] + 40, 
        prev_ego_pose['translation'][1] + 40
    ]
    
    # 渲染地图
    render_map_patch_compatible(
        nusc_map, 
        prev_box_coords, 
        ['drivable_area', 'walkway'], 
        ax=ax2
    )
    
    # 绘制自车位置（前一帧）
    ax2.plot(prev_ego_pose['translation'][0], prev_ego_pose['translation'][1], 
            'bo', markersize=10, label='Ego Vehicle')
    
    # 绘制自车方向（前一帧）
    # 提取旋转矩阵中的偏航角（简化处理）
    rotation = prev_ego_pose['rotation']
    if isinstance(rotation, list) and len(rotation) >= 4:
        # 四元数转换为偏航角
        from pyquaternion import Quaternion
        q = Quaternion(rotation)
        yaw = q.yaw_pitch_roll[0]
    else:
        # 使用简化的偏航角提取
        yaw = 0  # 默认值
    
    dx, dy = 5 * np.cos(yaw), 5 * np.sin(yaw)
    ax2.arrow(prev_ego_pose['translation'][0], prev_ego_pose['translation'][1],
             dx, dy, head_width=2, head_length=2, fc='b', ec='b')
    
    ax2.legend()
    ax2.grid(False)
    ax2.set_xlabel('X坐标')
    ax2.set_ylabel('Y坐标')
    
    plt.tight_layout()
    
    # 保存图像
    filename = f"map_comparison_{sample_token[:8]}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存
    
    print(f"图像已保存至: {save_path}")
    
    # 打印相关信息
    print(f"目标帧时间戳: {target_ego_pose['timestamp']}")
    print(f"前一帧时间戳: {prev_ego_pose['timestamp']}")
    print(f"时间差: {abs(target_ego_pose['timestamp'] - prev_ego_pose['timestamp'])/1e6:.2f} 秒")
    print(f"地图名称: {map_name}")
    
    # 返回保存路径
    return save_path

# 使用示例
if __name__ == "__main__":
    # 请替换为你的数据集路径
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
    
    # 指定目标sample的token
    target_sample_token = '0a0d6b8c2e884134a3b48df43d54c36a'
    
    # 保存目标帧和前一帧的地图
    save_sample_and_prev_map(nusc, target_sample_token, save_dir="./nuscenes_maps")