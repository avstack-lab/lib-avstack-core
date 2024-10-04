#!/usr/bin/env bash

set -e

MODEL=${1:?"missing arg 1 for MODEL"}
DATASET=${2:?"missing arg 2 for DATASET"}

# parse input model and dataset combination
if [ $MODEL = "unet" ]; then
    if [ $DATASET = "fov" ]; then
        config="../config/mmseg/unet_fov.py"
    else
        echo "Incompatible dataset passed for unet model!" 1>&2
        exit 64
    fi
else
    echo "Incompatible model passed!" 1>&2
    exit 64
fi

# set directory to mmsegmentation
MMSEGDIR="../../third_party/mmsegmentation"
MMSEGWORKDIR="../../third_party/mmsegmentation/work_dirs/${MODEL}_${DATASET}/"

# run distributed training
CUDA_VISIBLE_DEVICES="0,1"
CONFIG="$config"
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH=$MMSEGDIR:$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $MMSEGDIR/tools/train.py \
    $CONFIG \
    --work-dir $MMSEGWORKDIR \
    --launcher pytorch ${@:3}
