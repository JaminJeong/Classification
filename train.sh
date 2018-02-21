DATASET_DIR=./flowers_data_set
TRAIN_DIR=./exp_1
CHECKPOINT_PATH=./my_checkpoints/inception_v3.ckpt
SLIM_DIR=/home/dualmining/project/models/research/slim
#CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0 \
python ${SLIM_DIR}/train_image_classifier.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--dataset_name=flowers \
	--dataset_split_name=train \
	--model_name=inception_v3 \
	--checkpoint_path=${CHECKPOINT_PATH} \
	--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
	--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
