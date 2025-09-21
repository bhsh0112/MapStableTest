#!/usr/bin/env bash

# MapTR稳定性评估脚本 - NPZ格式
# 用于评估NPZ格式预测结果的稳定性指标

# 设置默认参数
CONFIG=${1:-"configs/default_config.py"}
PREDICTION_DIR=${2:-""}
OUTPUT_DIR=${3:-"outputs/stability_eval"}
DATA_ROOT=${4:-"/data/nuscenes"}
NUSC_VERSION=${5:-"v1.0-trainval"}
STABILITY_CLASSES=${6:-"divider,ped_crossing,boundary"}
STABILITY_INTERVAL=${7:-2}
LOCALIZATION_WEIGHT=${8:-0.5}

# 检查必需参数
if [ -z "$PREDICTION_DIR" ]; then
    echo "错误: 必须指定预测结果目录（包含若干 .npz 文件）"
    echo "用法: $0 <CONFIG> <PREDICTION_DIR> [OUTPUT_DIR] [DATA_ROOT] [NUSC_VERSION] [STABILITY_CLASSES] [STABILITY_INTERVAL] [LOCALIZATION_WEIGHT]"
    echo "示例: $0 configs/default_config.py results/npz_dir outputs/eval /data/nuscenes v1.0-trainval 'divider,ped_crossing,boundary' 1 0.5"
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$PREDICTION_DIR" ]; then
    echo "错误: 预测结果目录不存在: $PREDICTION_DIR"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

echo "MapTR稳定性评估 - NPZ格式"
echo "================================"
echo "配置文件: $CONFIG"
echo "预测目录: $PREDICTION_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "数据根目录: $DATA_ROOT"
echo "NuScenes版本: $NUSC_VERSION"
echo "稳定性类别: $STABILITY_CLASSES"
echo "帧间隔: $STABILITY_INTERVAL"
echo "位置权重: $LOCALIZATION_WEIGHT"
echo "================================"

# 设置Python路径
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# 执行稳定性评估
python3 main.py \
    --data-format npz \
    --config "$CONFIG" \
    --prediction-file "$PREDICTION_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --stability-interval $STABILITY_INTERVAL \
    --localization-weight $LOCALIZATION_WEIGHT \
    --pred-swap-xy \
    --pred-flip-y

echo "NPZ稳定性评估完成！"
