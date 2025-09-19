#!/usr/bin/env bash

# MapTR真值可视化脚本
# 用于可视化NuScenes数据集中的真值地图，生成稳定性分析图表

# 设置默认参数
SCENE_NAME=${1:-""}
DATA_ROOT=${2:-"/data/nuscenes"}
NUSC_VERSION=${3:-"v1.0-trainval"}
OUTPUT_DIR=${4:-"outputs/visualization"}
GIF_MODE=${5:-true}
GIF_SEPARATE=${6:-true}

# 检查必需参数
if [ -z "$SCENE_NAME" ]; then
    echo "错误: 必须指定场景名称"
    echo "用法: $0 <SCENE_NAME> [DATA_ROOT] [NUSC_VERSION] [OUTPUT_DIR] [GIF_MODE] [GIF_SEPARATE]"
    echo "示例: $0 scene-0001 /data/nuscenes v1.0-trainval outputs/vis true true"
    exit 1
fi

# 检查数据根目录是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 数据根目录不存在: $DATA_ROOT"
    exit 1
fi

echo "MapTR真值可视化"
echo "================================"
echo "场景名称: $SCENE_NAME"
echo "数据根目录: $DATA_ROOT"
echo "NuScenes版本: $NUSC_VERSION"
echo "输出目录: $OUTPUT_DIR"
echo "GIF模式: $GIF_MODE"
echo "分离GIF: $GIF_SEPARATE"
echo "================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置Python路径
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 构建可视化命令
VIS_CMD="python src/vis/vis_stability.py \
    --data-root \"$DATA_ROOT\" \
    --nusc-version \"$NUSC_VERSION\" \
    --scene-name \"$SCENE_NAME\" \
    --output-dir \"$OUTPUT_DIR\""

# 添加可选参数
if [ "$GIF_MODE" = true ]; then
    VIS_CMD="$VIS_CMD --gif"
fi

if [ "$GIF_SEPARATE" = true ]; then
    VIS_CMD="$VIS_CMD --gif-separate"
fi

# 执行可视化
echo "开始执行真值可视化..."
eval $VIS_CMD

if [ $? -eq 0 ]; then
    echo "真值可视化完成！"
    echo "输出文件保存在: $OUTPUT_DIR"
    echo "- scene_map.gif (地图可视化)"
    echo "- scene_six_view.gif (六视图)"
    echo "- map_frames/ (地图帧文件夹)"
    echo "- six_view_frames/ (六视图帧文件夹)"
else
    echo "真值可视化执行失败！"
    exit 1
fi
