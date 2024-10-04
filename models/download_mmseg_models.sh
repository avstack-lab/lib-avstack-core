#!/usr/bin/env bash

set -e

SAVEFOLDER=${1:-/data/$(whoami)/models}
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash
SAVEFOLDER="$SAVEFOLDER/mmseg"

THISDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Downloading models and saving to $SAVEFOLDER"


MMSEG_CKPT="${SAVEFOLDER}/checkpoints"
MMSEG_WKDIR="${SAVEFOLDER}/work_dirs"
mkdir -p "$MMSEG_CKPT"
mkdir -p "$MMSEG_WKDIR"

download_models () {
    MODEL_PATH="https://download.openmmlab.com/mmsegmentation"
    SUBFOLDER=$1  # Input 1: subfolder
    VERSION=$2    # Input 2: MMSEG version
    MODEL_TYPE=$3 # Input 3: model type
    MODEL=$4      # Input 4: model name
    OLDPATH=${4:-false}
    MNAME="${MODEL##*/}"
    if [ -f "${MMSEG_CKPT}/${SUBFOLDER}/${MNAME}" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading models"
        wget -P "${MMSEG_CKPT}/${SUBFOLDER}" "${MODEL_PATH}/${VERSION}/${MODEL_TYPE}/${MODEL}"

    fi
}

download_custom_models () {
    CUSTOM_MODEL_PATH="https://g-b0ef78.1d0d8d.03c0.data.globus.org/models/MMSEG"

    SUBFOLDER=$1  # Input 1: subfolder (e.g., "carla")
    MODEL_TYPE=$2 # Input 2: model type
    MODEL=$3      # Input 3: model name
    MNAME="${MODEL##*/}"
    if [ -f "${MMSEG_WKDIR}/${SUBFOLDER}/${MNAME}.pth" ]; then
        echo -e "$MODEL exists.\n"
    else 
        echo "Downloading model and configuration for $MODEL"
        MODPATH="$CUSTOM_MODEL_PATH"
        wget -P "${MMSEG_WKDIR}/${SUBFOLDER}" "${MODPATH}/work_dirs/${MODEL_TYPE}/${MODEL}.pth"
        wget -P "${MMSEG_WKDIR}/${SUBFOLDER}" "${MODPATH}/work_dirs/${MODEL_TYPE}/${MODEL}.py"
    fi
}

############################################
PSPNET_R50="pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
download_models "Cityscapes" "v0.5" "pspnet" "$PSPNET_R50"

############################################
DEEPLAB="deeplabv3plus_r50-d8_512x1024_40k_cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth"
download_models "Cityscapes" "v0.5" "deeplabv3plus" "$DEEPLAB"


echo "Adding symbolic link to mmseg directory"
ln -sfnT $(realpath "$MMSEG_CKPT") "$THISDIR/../third_party/mmsegmentation/checkpoints"
ln -sfnT $(realpath "$MMSEG_WKDIR") "$THISDIR/../third_party/mmsegmentation/work_dirs"