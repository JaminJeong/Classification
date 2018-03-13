SLIM_DIR=/home/dualmining/project/models/research/slim
CUDA_VISIBLE_DEVICES=5 \
python $SLIM_DIR/download_and_convert_data.py \
	--dataset_name=flowers \
	--dataset_dir=./tfrecords \
	--images_dir=./flowers/raw_data/flower_photos \
