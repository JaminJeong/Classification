#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0 \
python freezing.py \
    --log_dir=./exp_1 \
    --infer_flower=./inception_v3_inf_graph.pb \
    --model_flower=./exp_1/model.ckpt-1203 \

