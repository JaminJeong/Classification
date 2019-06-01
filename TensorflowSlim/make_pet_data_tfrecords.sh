HOME=/home/jamin
REPO_DIR=$HOME/projects/Code/Classification
export PYTHONPATH="${PYTHONPATH}:$REPO_DIR/TensorflowSlim"
CUDA_VISIBLE_DEVICES=0 \
python3 ./download_and_convert_data.py \
	--dataset_name=pet \
	--dataset_dir=$REPO_DIR/dataset \
#	--dataset_dir=your/repo/path \
