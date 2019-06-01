# Magic Hands image classification

### 1.1 데이터 수집

 - webcam 촬영 - OpenCV python 모듈을 이용하여 동영상을 이미지로 저장함.
 - 촬영 각도 : 수직 : 0 ~ 70 degree 수평 : 0 ~ 360 degree

### 2.1 Training

 - Dataset : 커머셜 can / pet 이미지
 - 학습모델 : MobileNet_v2
 - 총 학습 클래스 : 22종.
 - Optimizer : Adam, Exponential Learning
 - tfrecord 총 프레임 수 : 164,756 image files
 - Quantized model graph training. (cpu 모델)
 - Float model graph trianing. (gpu 모델)

### 2.2 Evaluation

 - Validation set : Train_set과 동일
 - Evaluation metric : binary accuracy
 - Threshold : 97 ~ 99%


### 3. 학습 데이터셋

- 시중에 유통되고 있는 음료상품들의 Fine-grained classification.
- Size, 상품별 구분.

학습 리스트 (2019. 03. 15)

	coke
	redbull
	cidar
	sprite
	narangd
	tropicana
	mountain_dew
	pepsi
	gatorade
	lipton
	fanta
	pet_coke
	pet_sprite
	pet_pepsi
	pet_gatorade
	pet_fanta_orange
	pet_fanta_pine
	pet_minute_maid
	pet_mountain_dew
	pet_lipton
	pet_tropicana_apple


### 4. 실행 순서

#### 4.0. 데이터 설정 변경

#### 4.1. tfrecords.sh
- tfrecords 생성

#### 4.2. train_can.sh
- 모델 학습 스크립트

#### 4.3. eval_can.sh
- 모델 평가 스크립트.

#### 4.4. export_graph.sh
 - mobilenet 그래프 생성.

#### 4.5. freezing.sh
 - model freezing ->  exported_graph + checkpoint(.ckpt) -> .pb 파일 변경

#### 4.6. tfconvert.sh
 - convert tflite file for mobile.

#### 4.7. inference_camera.sh
 - 테스트를 위한 pc 추정.

### 5. 데이터 설정 변경

dataset/can.py

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