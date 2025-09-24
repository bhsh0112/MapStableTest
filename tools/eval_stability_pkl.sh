#!/usr/bin/env bash

# MapTR稳定性评估脚本 - PKL格式
# 用于评估PKL格式预测结果的稳定性指标

# 设置默认参数
CONFIG=${1:-"configs/default_config.py"}
PREDICTION_FILE=${2:-""}
STABILITY_INTERVAL=${3:-2}
OUTPUT_DIR=${4:-"outputs/stability_eval"}
DATA_ROOT=${5:-"./data/nuscenes"}
NUSC_VERSION=${6:-"v1.0-trainval"}
STABILITY_CLASSES=${7:-"divider,ped_crossing,boundary"}

LOCALIZATION_WEIGHT=${8:-0.5}


# 检查必需参数
if [ -z "$PREDICTION_FILE" ]; then
    echo "错误: 必须指定预测结果文件"
    echo "用法: $0 <CONFIG> <PREDICTION_FILE> [OUTPUT_DIR] [DATA_ROOT] [NUSC_VERSION] [STABILITY_CLASSES] [STABILITY_INTERVAL] [LOCALIZATION_WEIGHT]"
    echo "示例: $0 configs/default_config.py results/pred.pkl outputs/eval /data/nuscenes v1.0-trainval 'divider,ped_crossing,boundary' 1 0.5"
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$PREDICTION_FILE" ]; then
    echo "错误: 预测结果文件不存在: $PREDICTION_FILE"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

echo "MapTR稳定性评估 - PKL格式"
echo "================================"
echo "配置文件: $CONFIG"
echo "预测文件: $PREDICTION_FILE"
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
python main.py \
    --data-format pkl \
    --config "$CONFIG" \
    --prediction-file "$PREDICTION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --stability-interval $STABILITY_INTERVAL \
    --localization-weight $LOCALIZATION_WEIGHT \
    --pred-swap-xy \
    --pred-flip-y

echo "PKL稳定性评估完成！"
