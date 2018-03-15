#!/bin/bash

IMAGES_DIR=./flower_photos
#NUM_EXAMPLES=
#NUM_EXAMPLES_VAL=

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=5 \
python build_image_data.py \
    --train_directory=$IMAGES_DIR/train \
    --validation_directory=$IMAGES_DIR/validation \
    --output_directory=./tfrecords \
    --labels_file=$IMAGES_DIR/labels.txt \
    --train_shards=64 \
    --validation_shards=16 \
    --num_threads=8 \
#   --num_examples=$NUM_EXAMPLES \
#   --num_examples_val=$NUM_EXAMPLES_VAL \
