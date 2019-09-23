from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras

tf.__version__

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 간단한 Sequential 모델을 반환합니다
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model


# 모델 객체를 만듭니다
model = create_model()
model.summary()
#
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # 체크포인트 콜백 만들기
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# model = create_model()
#
# model.fit(train_images, train_labels,  epochs = 10,
#           validation_data = (test_images,test_labels),
#           callbacks = [cp_callback])  # 훈련 단계에 콜백을 전달합니다


# 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 다섯 번째 에포크마다 가중치를 저장합니다
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)