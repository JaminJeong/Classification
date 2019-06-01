TFRECORD=../dataset/tfrecords
LOG_DIR=./log_pet_mobilenet

CUDA_VISIBLE_DEVICES=1 \
python3 eval_image_classifier.py \
  --batch_size=8 \
  --checkpoint_path=$LOG_DIR \
  --eval_dir=$LOG_DIR/eval/ \
  --dataset_name=pet \
  --dataset_split_name=train \
  --dataset_dir=${TFRECORD} \
  --model_name=mobilenet_v2 \
  --eval_image_size=224

# --quantize_delay=100
