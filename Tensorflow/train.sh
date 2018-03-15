#!/bin/bash

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=./log

# Where the dataset is saved to.
DATASET_DIR=../tfrecords

# Number of GPU
NUM_GPU=0
MAX_STEPS=10000
SAVE_STEPS=200
BATCH_SIZE=16
BATCH_SIZE_VAL=16
NUM_CLASSES=5
NUM_EXAMPLES=2937
NUM_EXAMPLES_VAL=733

# Run the training script.
CUDA_VISIBLE_DEVICES=$NUM_GPU \
python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --max_steps=$MAX_STEPS \
    --save_steps=$SAVE_STEPS \
    --batch_size=$BATCH_SIZE \
    --batch_size_val=$BATCH_SIZE_VAL \
    --num_classes=$NUM_CLASSES \
    --num_examples=$NUM_EXAMPLES \
    --num_examples_val=$NUM_EXAMPLES_VAL \
    --initial_learning_rate=0.0001 \
    --num_epochs_per_decay=200 \
    --learning_rate_decay_factor=0.96 \
    --optimizer='adam' \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --adam_epsilon=1.0E-08 \
    --pretrained_file_path=../pretrained_model/vgg_16.ckpt \
    --subset='validation' \
    --mode='train' \
    #--mode='inference' \

sleep 20


