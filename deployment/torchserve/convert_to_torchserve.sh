#!/usr/bin/env bash

set -e

MODEL=${1:-fasterrcnn}
DATASET=${2:-cityscapes}
MODEL_STORE=${3:-./model-store}


MM_BASE="../../third_party/mmdetection"
YAMLCFG="./model_config.yaml"


if [ "$MODEL" == "fasterrcnn" ]
then
    CONFIG="${MM_BASE}/configs/${DATASET}/faster-rcnn_r50_fpn_1x_cityscapes.py"
    CHKPT="${MM_BASE}/checkpoints/${DATASET}/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
else
    echo "Cannot parse model combination"
    exit 1
fi


poetry run python "${MM_BASE}/tools/deployment/mmdet2torchserve.py" ${CONFIG} ${CHKPT} ${YAMLCFG} \
    --output-folder "${MODEL_STORE}" \
    --model-name "${MODEL}_${DATASET}" \
    --force