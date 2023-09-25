#!/usr/bin/env bash

set -e

MODEL=${1:-cascade_rcnn}
DATASET=${2:-coco}
DEPCFG=${3:-detection_onnxruntime_dynamic.py}
RUNTIME=${4:-onnx}

MMDET_BASE="../../third_party/mmdetection"
MMDEP_BASE="../../third_party/mmdeploy"
DEPCFG="${MMDEP_BASE}/configs/mmdet/detection/${DEPCFG}"


# check if we've already converted this model
WORK_DIR="mmdeploy_models/${MODEL}_${DATASET}_${RUNTIME}"
if [ -d "$WORK_DIR" ]; then
    echo "Already converted this model -- saved to $WORK_DIR!"
    exit
fi

# Otherwise, parse the model configs/checkpoints
if [ "$MODEL" == "faster_rcnn" ]
then
    if [ "$DATASET" == "cityscapes" ]
    then
        DETCFG="${MMDET_BASE}/configs/${DATASET}/faster-rcnn_r50_fpn_1x_cityscapes.py"
        CHKPT="${MMDET_BASE}/checkpoints/${DATASET}/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
    elif [ "$DATASET" == "coco-person" ]
    then
        DETCFG="${MMDET_BASE}/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py"
        CHKPT="${MMDET_BASE}/checkpoints/${DATASET}/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
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
    --work-dir $WORK_DIR \
    --device cuda \
    --dump-info