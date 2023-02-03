#!/usr/bin/env bash

set -e

SAVEFOLDER=${1:-/data/$(whoami)/models}
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash
SAVEFOLDER="$SAVEFOLDER/mmdet3d"

echo "Downloading models and saving to $SAVEFOLDER"

THISDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODEL_PATH="https://download.openmmlab.com/mmdetection3d/v1.0.0_models"
OLD_MODELPATH="https://download.openmmlab.com/mmdetection3d/v0.1.0_models"

MMDET3D_CKPT="${SAVEFOLDER}/checkpoints"
MMDET3D_WKDIR="${SAVEFOLDER}/work_dirs"
mkdir -p "$MMDET3D_CKPT"
mkdir -p "$MMDET3D_WKDIR"

download_models () {
    SUBFOLDER=$1  # Input 1: subfolder
    MODEL_TYPE=$2 # Input 2: model type
    MODEL=$3     # Input 3: model name
    OLDPATH=${4:-false}
    MNAME="${MODEL##*/}"
    if [ -f "${MMDET3D_CKPT}/${SUBFOLDER}/${MNAME}" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading models"
        if [ "$OLDPATH" == true ]; then
            MODPATH="$OLD_MODELPATH"
        else
            MODPATH="$MODEL_PATH"
        fi
        wget -P "${MMDET3D_CKPT}/${SUBFOLDER}" "${MODPATH}/${MODEL_TYPE}/$MODEL"

    fi
}


KITTI_PILLARS="hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
download_models "kitti" "pointpillars" "$KITTI_PILLARS"
KITTI_PGD="pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth"
download_models "kitti" "pgd" "$KITTI_PGD"
NUIM_CMRCNN="cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim_20201009_124158-ad0540e3.pth"
download_models "nuimages" "nuimages_semseg" "$NUIM_CMRCNN" true
NUSC_PILLARS="hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth"
download_models "nuscenes" "fp16" "$NUSC_PILLARS" true
NUSC_PGD="pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth"
download_models "nuscenes" "pgd" "$NUSC_PGD"


echo "Adding symbolic link to mmdet3d directory"
ln -sfnT $(realpath "$MMDET3D_CKPT") "$THISDIR/../third_party/mmdetection3d/checkpoints"
ln -sfnT $(realpath "$MMDET3D_WKDIR") "$THISDIR/../third_party/mmdetection3d/work_dirs"