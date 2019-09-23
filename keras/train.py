# from keras import models
import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50

import os
from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.layers.normalization import BatchNormalization

import tensorflow as tf

class_num = 34

input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=input, include_top=False, weights='imagenet', pooling='max')
 
x = model.output
x = Dense(1024, name='fully', init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512, init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(class_num, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
model.summary()

train_dir = '../dataset/train/'
assert os.path.isdir(train_dir)
validation_dir = '../dataset/validation/'
assert os.path.isdir(validation_dir)

opt = optimizers.Adam(lr=1e-4)

model.compile(loss='categorical_crossentropy',
               optimizer=opt,
               metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

tb_hist = keras.callbacks.TensorBoard(log_dir='./graph',
                                      histogram_freq=0,
                                      write_graph=True,
                                      write_images=True)

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=False,
                                              verbose=1,
                                              period=5)

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
# manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
# ckpt.restore(manager.latest_checkpoint)
# if manager.latest_checkpoint:
#   print("Restored from {}".format(manager.latest_checkpoint))
# else:
#   print("Initializing from scratch.")


# if os.path.isdir(checkpoint_dir) and len(os.listdir(checkpoint_dir)) != 0:
#     latest = tf.train.latest_checkpoint(checkpoint_dir)
#     print("load_weights : ", latest)

# model.load_weights("training_1/cp-0010.ckpt")

model.save_weights(checkpoint_path.format(epoch=0))

hist = model.fit_generator(
        train_generator,
        steps_per_epoch=75,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=75,
        callbacks=[tb_hist, cp_callback])
        # initial_epoch=10)

model.save('pet.h5')

