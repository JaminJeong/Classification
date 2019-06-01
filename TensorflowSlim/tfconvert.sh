MODEL_INPUT_GRAPH=input
MODEL_OUTPUT_GRAPH=MobilenetV2/Predictions/Reshape_1

tflite_convert \
  --output_file=./pb_file/cl_model.tflite \
  --graph_def_file=./pb_file/frozen_cl_model.pb \
  --input_arrays=$MODEL_INPUT_GRAPH \
  --output_arrays=$MODEL_OUTPUT_GRAPH \