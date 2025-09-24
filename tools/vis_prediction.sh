#!/usr/bin/env bash

# MapTR预测结果可视化脚本
# 用于可视化PKL格式的预测结果，生成稳定性分析图表

# 设置默认参数
SCENE_NAME=${1:-""}
PREDICTION_FILE=${2:-""}
DATA_ROOT=${3:-"./data/nuscenes"}
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

# 从预测文件名推导“配置名”（无扩展名），用于组织输出目录
CONFIG_NAME=$(basename "$PREDICTION_FILE")
CONFIG_NAME="${CONFIG_NAME%.*}"

# 如果 SCENE_NAME 为 all，则遍历所有场景并逐一可视化
if [ "$SCENE_NAME" = "all" ]; then
    echo "检测到 all，开始遍历所有场景..."
    # 若指定版本子目录不存在，尝试自动回退到 v1.0-mini
    if [ ! -d "$DATA_ROOT/$NUSC_VERSION" ]; then
        if [ -d "$DATA_ROOT/v1.0-mini" ]; then
            echo "未找到 $DATA_ROOT/$NUSC_VERSION ，自动回退为 v1.0-mini"
            NUSC_VERSION="v1.0-mini"
        else
            echo "错误: 未找到版本目录: $DATA_ROOT/$NUSC_VERSION 且无 v1.0-mini 可用"
            exit 1
        fi
    fi

    # 通过Python从NuScenes读取所有scene名称（显式传入环境变量）
    SCENES=$(DATA_ROOT="$DATA_ROOT" NUSC_VERSION="$NUSC_VERSION" python - <<'PY'
from nuscenes.nuscenes import NuScenes
import os
data_root = os.environ.get('DATA_ROOT', '')
version = os.environ.get('NUSC_VERSION', 'v1.0-trainval')
nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
for scene in nusc.scene:
    print(scene['name'])
PY
    )

    IFS=$'\n' read -rd '' -a SCENE_ARRAY <<<"$SCENES" || true
    if [ ${#SCENE_ARRAY[@]} -eq 0 ]; then
        echo "未能获取到任何场景名称，请检查数据路径与版本。"; exit 1
    fi

    # 计算脚本自身的绝对路径，避免 PATH 问题
    SELF_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    SELF_SCRIPT="$SELF_SCRIPT_DIR/$(basename "$0")"
    echo "SCENE_ARRAY: ${SCENE_ARRAY[@]}"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    for S in "${SCENE_ARRAY[@]}"; do
        SUB_OUT="$OUTPUT_DIR"
        mkdir -p "$SUB_OUT"
        echo "\n>>> 处理场景: $S，输出到: $SUB_OUT"

        echo "运行: $SELF_SCRIPT \"$S\" \"$PREDICTION_FILE\" \"$DATA_ROOT\" \"$NUSC_VERSION\" \"$SUB_OUT\" $GIF_MODE $GIF_SEPARATE $PRED_SWAP_XY $PRED_FLIP_Y $PRED_ROTATE_DEG"
        DATA_ROOT="$DATA_ROOT" NUSC_VERSION="$NUSC_VERSION" \
        "$SELF_SCRIPT" "$S" "$PREDICTION_FILE" "$DATA_ROOT" "$NUSC_VERSION" "$SUB_OUT" $GIF_MODE $GIF_SEPARATE $PRED_SWAP_XY $PRED_FLIP_Y $PRED_ROTATE_DEG
        if [ $? -ne 0 ]; then
            echo "场景 $S 可视化失败，继续下一个场景..."
        fi
    done

    echo "所有场景可视化完成。"
    exit 0
fi

# 创建输出目录（单场景）：<输出根>/<配置名>/<场景名>
OUTPUT_DIR_FINAL="$OUTPUT_DIR/$CONFIG_NAME/$SCENE_NAME"
mkdir -p "$OUTPUT_DIR_FINAL"

# 设置Python路径
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 构建可视化命令
VIS_CMD="python src/vis/vis_stability.py \
    --pred \"$PREDICTION_FILE\" \
    --data-root \"$DATA_ROOT\" \
    --nusc-version \"$NUSC_VERSION\" \
    --scene-name \"$SCENE_NAME\" \
    --out-dir \"$OUTPUT_DIR_FINAL\""

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
