# Pretrained Checkpoint

### script
```bash
bash download_pretrained_ckpt.sh
```
 - download_pretrained_ckpt.sh
    - download ckpt files and extract tar

### Folder Structure
```bash
├── README.md
├── download_pretrained_ckpt.py
├── download_pretrained_ckpt.sh
└── mobilenet_v2_1.0_224
    ├── mobilenet_v2_1.0_224.ckpt.data-00000-of-00001
    ├── mobilenet_v2_1.0_224.ckpt.index
    ├── mobilenet_v2_1.0_224.ckpt.meta
    ├── mobilenet_v2_1.0_224.tflite
    ├── mobilenet_v2_1.0_224.tgz
    ├── mobilenet_v2_1.0_224_eval.pbtxt
    ├── mobilenet_v2_1.0_224_frozen.pb
    └── mobilenet_v2_1.0_224_info.txt
```

### Requirements
 - wget

### Reference
 - https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
