# Run evaluation.
CUDA_VISIBLE_DEVICES=1 \
python ./eval_image_classifier.py \
  --checkpoint_path=./log \
  --eval_dir=./log \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=./tfrecords \
  --batch_size=4 \
  --model_name=inception_v3 \
