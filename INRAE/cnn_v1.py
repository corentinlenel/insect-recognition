from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

path_to_db = '/home/corentin/Documents/INRAE/V1'

if len(sys.argv) > 2:
    print("Trop de paramètre en entrée")
    sys.exit(1)
elif len(sys.argv) > 1:
    EPOCHS=int(sys.argv[1])
else:
    EPOCHS=50 

NUM_OF_CLASS = 2
BATCH_SIZE = 16
IMG_HEIGHT = 150
IMG_WIDTH = 150

PATH = os.path.join(path_to_db)

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
models_dir = os.path.join(PATH, 'models')

num_models = len(os.listdir(models_dir))
model_file = models_dir +  '/model_' + str(num_models + 1) + '.h5'

# Init our model 

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# First layer setup

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second layer setup

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third layer setup

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary() 

train_image_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')



STEP_SIZE_TRAIN=train_data_gen.n//BATCH_SIZE
STEP_SIZE_VALID=val_data_gen.n//BATCH_SIZE

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_data_gen,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS
)

model.save(model_file)

score = model.evaluate(val_data_gen, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Model be saved in the directory ', model_file)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()