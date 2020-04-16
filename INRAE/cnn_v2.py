from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, MaxPool2D, Activation, ZeroPadding2D, Convolution2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

path_to_db = '/home/corentin/Documents/INRAE/V2/'


if len(sys.argv) > 2:
    print("Trop de paramètre en entrée")
    sys.exit(1)
elif len(sys.argv) > 1:
    EPOCHS=int(sys.argv[1])
else:
    EPOCHS=50 

EPOCHS=10 
BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150

PATH = os.path.join(path_to_db)
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
models_dir = os.path.join(PATH, 'models')

NUM_OF_CLASS=len(os.listdir(validation_dir))

num_models = len(os.listdir(models_dir))
model_file = models_dir +  '/model_' + str(num_models + 1) + '.h5'

# Init our model 

model = Sequential()
model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)))
model.add(MaxPooling2D())

# First layer 

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())

# Second layer 

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())

# Third layer 

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(NUM_OF_CLASS, activation='softmax'))

opt= RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() 

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

STEP_SIZE_TRAIN=train_data_gen.n//BATCH_SIZE
STEP_SIZE_VALID=val_data_gen.n//BATCH_SIZE

history = model.fit(
    train_data_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=STEP_SIZE_VALID
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