# Classification Using Tensorflow Silm 

### How to training

#### 4.0. 데이터 설정 변경

#### 4.1. tfrecords.sh
- create tfrecords files

#### 4.2. train.sh
 - gpu

#### 4.3. eval.sh
 - cpu

#### 4.4. export_graph.sh
 - crate graph.pb 

#### 4.5. freezing.sh
 - model freezing ->  exported_graph + checkpoint(.ckpt) -> .pb

#### 4.6. tfconvert.sh
 - convert tflite file.

#### 4.7. inference_camera.sh
 - inferece using camera for test

### 5. 데이터 설정 변경

dataset/pet.py

```python
_FILE_PATTERN = 'cans_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 162065, 'test': 162065}

_NUM_CLASSES = 21
```

dataset/convert_can.py

```python    
_NUM_VALIDATION = 300 // train validation 을 나눌 비율
```

preprocessing/inception_preprocessing.py
```python
def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.6,
                                aspect_ratio_range=(0.4, 1.0), // 가로 세로 비율
                                area_range=(0.8, 1.0), // crop 하는 이미지 비율
                                max_attempts=100,
                                scope=None):

def preprocess_for_eval(image, height, width,
                        central_fraction=1.0, scope=None):
```

### 6. system configuration
- Ubuntu : 16.04
- nvidia-driver : 387.26(or 390)
- cuda toolkit library : 9.0
- cudnn : 7.1.3
- Python : 3.5.2
- Tensorflow : 1.12.0
- OpenCV : 3.4.0

### 7. reference
 - https://github.com/tensorflow/models/tree/master/research/slim
 - https://www.tensorflow.org/lite/convert/cmdline_examples
 - https://www.tensorflow.org/lite/convert
 - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/cmdline_examples.md
 - https://arxiv.org/abs/1801.04381