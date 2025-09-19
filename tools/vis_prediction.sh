#!/usr/bin/env bash

# MapTR预测结果可视化脚本
# 用于可视化PKL格式的预测结果，生成稳定性分析图表

# 设置默认参数
SCENE_NAME=${1:-""}
PREDICTION_FILE=${2:-""}
DATA_ROOT=${3:-"/data/nuscenes"}
NUSC_VERSION=${4:-"v1.0-trainval"}
OUTPUT_DIR=${5:-"outputs/visualization"}
GIF_MODE=${6:-true}
GIF_SEPARATE=${7:-true}
PRED_SWAP_XY=${8:-true}
PRED_FLIP_Y=${9:-true}
PRED_ROTATE_DEG=${10:-0.0}

# 检查必需参数
if [ -z "$SCENE_NAME" ] || [ -z "$PREDICTION_FILE" ]; then
    echo "错误: 必须指定场景名称和预测结果文件"
    echo "用法: $0 <SCENE_NAME> <PREDICTION_FILE> [DATA_ROOT] [NUSC_VERSION] [OUTPUT_DIR] [GIF_MODE] [GIF_SEPARATE] [PRED_SWAP_XY] [PRED_FLIP_Y] [PRED_ROTATE_DEG]"
    echo "示例: $0 scene-0001 results/pred.pkl /data/nuscenes v1.0-trainval outputs/vis true true true true 0.0"
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$PREDICTION_FILE" ]; then
    echo "错误: 预测结果文件不存在: $PREDICTION_FILE"
    exit 1
fi

# 检查数据根目录是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 数据根目录不存在: $DATA_ROOT"
    exit 1
fi

echo "MapTR预测结果可视化"
echo "================================"
echo "场景名称: $SCENE_NAME"
echo "预测文件: $PREDICTION_FILE"
echo "数据根目录: $DATA_ROOT"
echo "NuScenes版本: $NUSC_VERSION"
echo "输出目录: $OUTPUT_DIR"
echo "GIF模式: $GIF_MODE"
echo "分离GIF: $GIF_SEPARATE"
echo "交换XY: $PRED_SWAP_XY"
echo "翻转Y轴: $PRED_FLIP_Y"
echo "旋转角度: $PRED_ROTATE_DEG"
echo "================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置Python路径
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 构建可视化命令
VIS_CMD="python src/vis/vis_stability.py \
    --pred \"$PREDICTION_FILE\" \
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

if [ "$PRED_SWAP_XY" = true ]; then
    VIS_CMD="$VIS_CMD --pred-swap-xy"
fi

if [ "$PRED_FLIP_Y" = true ]; then
    VIS_CMD="$VIS_CMD --pred-flip-y"
fi

if [ "$PRED_ROTATE_DEG" != "0.0" ]; then
    VIS_CMD="$VIS_CMD --pred-rotate-deg $PRED_ROTATE_DEG"
fi

# 执行可视化
echo "开始执行可视化..."
eval $VIS_CMD

if [ $? -eq 0 ]; then
    echo "预测结果可视化完成！"
    echo "输出文件保存在: $OUTPUT_DIR"
    echo "- scene_map.gif (地图可视化)"
    echo "- scene_six_view.gif (六视图)"
    echo "- map_frames/ (地图帧文件夹)"
    echo "- six_view_frames/ (六视图帧文件夹)"
else
    echo "可视化执行失败！"
    exit 1
fi
