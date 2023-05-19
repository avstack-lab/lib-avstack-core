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
    MODEL_PATH="https://download.openmmlab.com/mmdetection"
    SUBFOLDER=$1  # Input 1: subfolder
    VERSION=$2    # Input 2: mmdet version
    MODEL_TYPE=$3 # Input 3: model type
    MODEL=$4      # Input 4: model name
    OLDPATH=${4:-false}
    MNAME="${MODEL##*/}"
    if [ -f "${MMDET_CKPT}/${SUBFOLDER}/${MNAME}" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading models"
        wget -P "${MMDET_CKPT}/${SUBFOLDER}" "${MODEL_PATH}/${VERSION}/${MODEL_TYPE}/${MODEL}"

    fi
}

download_custom_models () {
    CUSTOM_MODEL_PATH="https://g-b0ef78.1d0d8d.03c0.data.globus.org/models/mmdet"

    SUBFOLDER=$1  # Input 1: subfolder (e.g., "carla")
    MODEL_TYPE=$2 # Input 2: model type
    MODEL=$3      # Input 3: model name
    MNAME="${MODEL##*/}"
    if [ -f "${MMDET_WKDIR}/${SUBFOLDER}/${MNAME}.pth" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading model and configuration for $MODEL"
        MODPATH="$CUSTOM_MODEL_PATH"
        wget -P "${MMDET_WKDIR}/${SUBFOLDER}" "${MODPATH}/work_dirs/${MODEL_TYPE}/${MODEL}.pth"
        wget -P "${MMDET_WKDIR}/${SUBFOLDER}" "${MODPATH}/work_dirs/${MODEL_TYPE}/${MODEL}.py"
    fi
}

COCOPERSON_FRCNN="faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
download_models "coco-person" "v2.0" "faster_rcnn" "$COCOPERSON_FRCNN"
CITYSCAPES_FRCNN="faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
download_models "cityscapes" "v2.0" "cityscapes" "$CITYSCAPES_FRCNN"
CITYSCAPES_MRCNN="mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth"
download_models "cityscapes" "v2.0" "cityscapes" "$CITYSCAPES_MRCNN"

RTMDET_M="rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
download_models "rtmdet" "v3.0" "rtmdet" "$RTMDET_M"

CARLA_FRCNN="faster_rcnn_r50_fpn_1x_carla"
download_custom_models "carla" "carla" "$CARLA_FRCNN"
CARLA_FRCNN_INF="faster_rcnn_r50_fpn_1x_carla_infrastructure"
download_custom_models "carla" "carla" "$CARLA_FRCNN_INF"

NUSC_FRCNN="faster_rcnn_r50_fpn_1x_nuscenes"
download_custom_models "nuscenes" "nuscenes" "$NUSC_FRCNN"


echo "Adding symbolic link to mmdet directory"
ln -sfnT $(realpath "$MMDET_CKPT") "$THISDIR/../third_party/mmdetection/checkpoints"
ln -sfnT $(realpath "$MMDET_WKDIR") "$THISDIR/../third_party/mmdetection/work_dirs"
