LOG_DIR=./log_pet_mobilenet
PB_DIR=./pb_file
MODEL_OUTPUT_GRAPH=MobilenetV2/Predictions/Reshape_1

if [ ! -d $PB_DIR ]; then
    mkdir $PB_DIR
fi

CUDA_VISIBLE_DEVICES=0 \
freeze_graph --input_graph=./export_mobilenet.pb \
  --input_checkpoint=$LOG_DIR/model.ckpt-10022 \
  --input_binary=true \
  --output_graph=$PB_DIR/frozen_cl_model.pb \
  --output_node_names=$MODEL_OUTPUT_GRAPH