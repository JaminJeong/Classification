# from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50

import os
from keras.layers import Input

input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=input, include_top=True, weights=None, pooling='max')
model.summary()

train_dir = '../dataset/train/'
assert os.path.isdir(train_dir)
validation_dir = '../dataset/validation/'
assert os.path.isdir(validation_dir)

model.compile(loss='categorical_crossentropy',
               optimizer=optimizers.Adam(lr=1e-4),
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


history = model.fit_generator(
        train_generator,
        steps_per_epoch=75,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=75)

model.save('pet.h5')