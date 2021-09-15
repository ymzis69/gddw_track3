#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# python $(dirname "$0")/test.py $CONFIG work_dirs/detectors_cascade_rcnn_r50_12_coco_label_smoothing_0.5/epoch_12.pth --format-only
