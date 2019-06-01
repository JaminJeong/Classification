TFRECORD=../dataset/tfrecords
PRET_CKPT=../pretrained_ckpt/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt
LOG_DIR=./log_pet_mobilenet

if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=0 \
python ./train_image_classifier.py \
     --train_dir=$LOG_DIR \
     --dataset_dir=${TFRECORD} \
     --dataset_name=pet \
     --dataset_split_name=train \
     --model_name=mobilenet_v2 \
     --checkpoint_path=${PRET_CKPT} \
     --checkpoint_exclude_scopes='MobilenetV2/Logits' \
     --save_summaries_sec=100  \
     --save_interval_secs=100  \
     --preprocessing_name=mobilenet_v2 \
     --batch_size=32 \
     --learning_rate=0.001 \
     --learning_rate_decay_type=fixed \
     --train_image_size=224 \

#    --model_name=inception_v3 \
#    --log_every_n_steps=100 \
#    --max_number_of_steps=20000 \
#    --optimizer=adam \
#    --weight_decay=0.00004
#    --ignore_missing_vars=True \
#    --checkpoint_exclude_scopes=final_layer,aux_11/aux_logits/FC \
#    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
