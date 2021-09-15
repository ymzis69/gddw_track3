#!/usr/bin/env bash

# create train json
python data_process/2json_train.py

CUDA_VISIBLE_DEVICES=0 PORT=1111 ./tools/dist_train.sh configs/detectoRS_r101.py 1

python data_process/model_dist.py
