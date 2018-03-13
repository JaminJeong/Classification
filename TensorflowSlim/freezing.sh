#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0 \
python freezing.py \
    --log_dir=./log \
    --model_name=flowers \
    --infer_pb_file=./inception_v3_inf_graph.pb \
    --model_log_ckpt=./log/model.ckpt-xxxx \
    #--model_flower=./log/model.ckpt-1234 \

