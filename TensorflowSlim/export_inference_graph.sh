CUDA_VISIBLE_DEVICES=0 \
python3 export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v2 \
  --output_file=./export_mobilenet.pb \
  --dataset_name=pet \
  --image_size=224 \
