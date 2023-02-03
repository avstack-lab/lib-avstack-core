#!/usr/bin/env bash

set -e

SAVEFOLDER=${1:-/data/$(whoami)/models}
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash
SAVEFOLDER="$SAVEFOLDER/mmdet"

THISDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading models and saving to $SAVEFOLDER"

MODEL_PATH="https://download.openmmlab.com/mmdetection/v2.0"

MMDET_CKPT="${SAVEFOLDER}/checkpoints"
MMDET_WKDIR="${SAVEFOLDER}/work_dirs"
mkdir -p "$MMDET_CKPT"
mkdir -p "$MMDET_WKDIR"

download_models () {
    SUBFOLDER=$1  # Input 1: subfolder
    MODEL_TYPE=$2 # Input 2: model type
    MODEL=$3     # Input 3: model name
    OLDPATH=${4:-false}
    MNAME="${MODEL##*/}"
    if [ -f "${MMDET_CKPT}/${SUBFOLDER}/${MNAME}" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading models"
        wget -P "${MMDET_CKPT}/${SUBFOLDER}" "${MODEL_PATH}/${MODEL_TYPE}/$MODEL"

    fi
}

COCOPERSON_FRCNN="faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
download_models "coco-person" "faster_rcnn" "$COCOPERSON_FRCNN"
CITYSCAPES_FRCNN="faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
download_models "cityscapes" "cityscapes" "$CITYSCAPES_FRCNN"
CITYSCAPES_MRCNN="mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth"
download_models "cityscapes" "cityscapes" "$CITYSCAPES_MRCNN"

echo "Adding symbolic link to mmdet directory"
ln -sf $(realpath "$MMDET_CKPT") "$THISDIR/../third_party/mmdetection/checkpoints"
ln -sf $(realpath "$MMDET_WKDIR") "$THISDIR/../third_party/mmdetection/work_dirs"
