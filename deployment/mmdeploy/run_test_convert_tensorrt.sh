#!/usr/bin/env bash

set -e

MODEL=${1:-cascade_rcnn}
DATASET=${2:-coco}
DEPCFG="detection_tensorrt_dynamic-320x320-1344x1344.py"
bash _base_convert_model.sh $MODEL $DATASET $DEPCFG tensorrt