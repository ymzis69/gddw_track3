#!/usr/bin/env bash

# create test json
python data_process/testb2json.py

CUDA_VISIBLE_DEVICES=0 python ./tools/test.py ./configs/detectoRS_r101.py ../user_data/model_data/epoch_12_dist.pth --format-only
