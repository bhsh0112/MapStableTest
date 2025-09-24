#!/usr/bin/env bash
SCENE=$1
python src/vis/vis_stability.py --data-root data/nuscenes --nusc-version v1.0-trainval --scene-name $SCENE --pred-swap-xy --pred-flip-y --gif --gif-separate