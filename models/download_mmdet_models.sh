#!/usr/bin/env bash

set -e

SAVEFOLDER=${1:-/data/$(whoami)/models}
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash
SAVEFOLDER="$SAVEFOLDER/mmdet"

THISDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading models and saving to $SAVEFOLDER"


MMDET_CKPT="${SAVEFOLDER}/checkpoints"
MMDET_WKDIR="${SAVEFOLDER}/work_dirs"
mkdir -p "$MMDET_CKPT"
mkdir -p "$MMDET_WKDIR"

download_models () {
    MODEL_PATH="https://download.openmmlab.com/mmdetection/v2.0"
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

download_aws_models () {
    AWS_MODEL_PATH="https://avstack-public-data.s3.amazonaws.com/models/mmdet"
    SUBFOLDER=$1  # Input 1: subfolder (e.g., "carla")
    MODEL_TYPE=$2 # Input 2: model type
    MODEL=$3      # Input 3: model name
    MNAME="${MODEL##*/}"
    if [ -f "${MMDET_WKDIR}/${SUBFOLDER}/${MNAME}.pth" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading model and configuration for $MODEL"
        MODPATH="$AWS_MODEL_PATH"
        wget -P "${MMDET_WKDIR}/${SUBFOLDER}" "${MODPATH}/${MODEL_TYPE}/${MODEL}.pth"
        wget -P "${MMDET_WKDIR}/${SUBFOLDER}" "${MODPATH}/${MODEL_TYPE}/${MODEL}.py"
    fi
}

COCOPERSON_FRCNN="faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
download_models "coco-person" "faster_rcnn" "$COCOPERSON_FRCNN"
CITYSCAPES_FRCNN="faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
download_models "cityscapes" "cityscapes" "$CITYSCAPES_FRCNN"
CITYSCAPES_MRCNN="mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth"
download_models "cityscapes" "cityscapes" "$CITYSCAPES_MRCNN"

CARLA_FRCNN="faster_rcnn_r50_fpn_1x_carla"
download_aws_models "carla" "carla" "$CARLA_FRCNN"
CARLA_FRCNN_INF="faster_rcnn_r50_fpn_1x_carla_infrastructure"
download_aws_models "carla" "carla" "$CARLA_FRCNN_INF"

NUSC_FRCNN="faster_rcnn_r50_fpn_1x_nuscenes"
download_aws_models "nuscenes" "nuscenes" "$NUSC_FRCNN"


echo "Adding symbolic link to mmdet directory"
ln -sf $(realpath "$MMDET_CKPT") "$THISDIR/../third_party/mmdetection/checkpoints"
ln -sf $(realpath "$MMDET_WKDIR") "$THISDIR/../third_party/mmdetection/work_dirs"
