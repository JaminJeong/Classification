PRETRAINED_CHECKPOINT_DIR=../pretrained_model/nasnet-a_large_04_10_2017

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi

CUDA_VISIBLE_DEVICES=0 \
python3 ./train_image_classifier.py \
     --train_dir=./log \
     --dataset_dir=../tfrecords \
     --dataset_name=flower \
     --dataset_split_name=train \
     --model_name=inception_v3 \
     --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/model.ckpt \
     --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
     --save_summaries_sec=100  \
     --save_interval_secs=100  \
     --preprocessing_name=inception_v3 \
     --batch_size=4 \
     --learning_rate=0.001 \
     --learning_rate_decay_type=fixed \
#    --log_every_n_steps=100 \
#    --max_number_of_steps=20000 \
#    --optimizer=rmsprop \
#    --weight_decay=0.00004
#    --ignore_missing_vars=True \
#    --checkpoint_exclude_scopes=final_layer,aux_11/aux_logits/FC \
#    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
