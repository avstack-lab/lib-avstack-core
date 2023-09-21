#!/usr/bin/env bash

set -e

MODEL=${1:-cascade_rcnn}
DATASET=${2:-coco}

MMDET_BASE="../../third_party/mmdetection"
MMDEP_BASE="../../third_party/mmdeploy"


if [ "$MODEL" == "faster_rcnn" ]
then
    if [ "$DATASET" == "cityscapes" ]
    then
        DETCFG="${MMDET_BASE}/configs/${DATASET}/faster-rcnn_r50_fpn_1x_cityscapes.py"
        CHKPT="${MMDET_BASE}/checkpoints/${DATASET}/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
        DEPCFG="${MMDEP_BASE}/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py"
    else
        echo "Cannot parse dataset combination"
        exit 1
    fi
elif [ "$MODEL" == "cascade_rcnn" ]
then
    if [ "$DATASET" == "coco" ]
    then
        DETCFG="${MMDET_BASE}/configs/${MODEL}/cascade-rcnn_r50_fpn_1x_coco.py"
        CHKPT="${MMDET_BASE}/checkpoints/${DATASET}/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"
        DEPCFG="${MMDEP_BASE}/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py"
    else
        echo "Cannot parse dataset combination"
        exit 1
    fi
else
    echo "Cannot parse model combination"
    exit 1
fi


# run the command to start model conversion
poetry run python "${MMDEP_BASE}/tools/deploy.py" \
    $DEPCFG \
    $DETCFG \
    $CHKPT \
    ${MMDET_BASE}/demo/demo.jpg \
    --work-dir "mmdeploy_models/${MODEL}_${DATASET}" \
    --device cuda \
    --dump-info