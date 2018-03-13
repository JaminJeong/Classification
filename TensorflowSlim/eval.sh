# Run evaluation.
CUDA_VISIBLE_DEVICES=0 \
python3 ./eval_image_classifier.py \
  --checkpoint_path=./log \
  --eval_dir=./log \
  --dataset_name=car \
  --dataset_split_name=validation \
  --dataset_dir=../tfrecords \
  --batch_size=4 \
  --model_name=inception_v3 \
