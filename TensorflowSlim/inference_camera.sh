CUDA_VISIBLE_DEVICES=0 \
python3 inference_camera.py \
    --pb_file=./pb_file/frozen_cl_model.pb \
    --label_path=./data/label_map.txt \
