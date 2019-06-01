
#source ./environment_setting.sh
#CUDA_VISIBLE_DEVICES=0 \
#python3 freezing.py \
#    --log_dir=./pb_file \
#    --infer_car=./export_mobilenet.pb \
#    --model_car=./logs/can_log_quant2/model.ckpt-567831 \
#    --model_name='cl_model' \
#    --model_input_graph=$MODEL_INPUT_GRAPH \
#    --model_output_graph=$MODEL_OUTPUT_GRAPH \


CUDA_VISIBLE_DEVICES=0 \
freeze_graph --input_graph=./export_mobilenet.pb \
  --input_checkpoint=./logs/can_log_quant2/model.ckpt-567831 \
  --input_binary=true \
  --output_graph=./pb_file/frozen_cl_model.pb \
  --output_node_names=MobilenetV2/Predictions/Reshape_1