#!/usr/bin/env bash
SCENE=$1
PKL=$2
python tools/vis_stability.py --pred $PKL --data-root data/nuscenes --nusc-version v1.0-trainval --scene-name $SCENE --pred-swap-xy --pred-flip-y --gif --gif-separate