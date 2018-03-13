CUDA_VISIBLE_DEVICES=0 \
python ./export_inference_graph.py \
	--model_name=inception_v3 \
	--output_file=./export_output_graph.py \
	--dataset_name=flowers \
