# Classification Using Tensorflow Silm 

### How to training

#### 1. tfrecords.sh
- create tfrecords files

#### 2. train.sh
 - gpu

#### 3. eval.sh
 - cpu

#### 4. export_graph.sh
 - create graph.pb 

#### 5. freezing.sh
 - model freezing ->  exported_graph + checkpoint(.ckpt) -> .pb

#### 6. tfconvert.sh
 - convert tflite file.

#### 7. inference_camera.sh
 - inferece using camera for test

### 

dataset/pet.py

```python
_FILE_PATTERN = 'pet_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 6069, 'test': 675}

_NUM_CLASSES = 34
```

preprocessing/inception_preprocessing.py
```python
def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.6,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.8, 1.0),
                                max_attempts=100,
                                scope=None):

def preprocess_for_eval(image, height, width,
                        central_fraction=1.0, scope=None):
```
### 7. reference
 - https://github.com/tensorflow/models/tree/master/research/slim
 - https://www.tensorflow.org/lite/convert/cmdline_examples
 - https://www.tensorflow.org/lite/convert
 - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/cmdline_examples.md
 - https://arxiv.org/abs/1801.04381
