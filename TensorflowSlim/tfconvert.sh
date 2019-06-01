source ./environment_setting.sh

tflite_convert \
  --output_file=$RUN_PATH/MagicHands/classification/pb_file/cl_model.tflite \
  --graph_def_file=$RUN_PATH/MagicHands/classification/pb_file/frozen_cl_model.pb \
  --input_arrays=$MODEL_INPUT_GRAPH \
  --output_arrays=$MODEL_OUTPUT_GRAPH \

