#!/usr/bin/env bash

set -e

MODEL=${1:-cascade_rcnn}
DATASET=${2:-coco}
DEPCFG="detection_onnxruntime_dynamic.py"
bash _base_convert_model.sh $MODEL $DATASET $DEPCFG onnx