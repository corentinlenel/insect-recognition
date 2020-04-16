from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

path_to_db = '/home/corentin/Documents/IRSTA/V2'
PATH = os.path.join(path_to_db)

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_eucnemidae_hylis_foveicollis_dir = os.path.join(train_dir, 'Eucnemidae_Hylis_foveicollis')  
train_eucnemidae_hylis_olexai_dir = os.path.join(train_dir, 'Eucnemidae_Hylis_olexai') 
train_eucnemidae_microrhagus_lepidus_dir = os.path.join(train_dir, 'Eucnemidae_Microrhagus_lepidus')  
train_eucnemidae_microrhagus_pygmaeus_dir = os.path.join(train_dir, 'Eucnemidae_Microrhagus_pygmaeus')  
train_salpingidae_rabocerus_foveolatus_dir = os.path.join(train_dir, 'Salpingidae_Rabocerus_foveolatus')  
train_salpingidae_salpingus_palnirostris_dir = os.path.join(train_dir, 'Salpingidae_Salpingus_palnirostris')

validation_eucnemidae_hylis_foveicollis_dir = os.path.join(validation_dir, 'Eucnemidae_Hylis_foveicollis')  
validation_eucnemidae_hylis_olexai_dir = os.path.join(validation_dir, 'Eucnemidae_Hylis_olexai')
validation_eucnemidae_microrhagus_lepidus_dir = os.path.join(validation_dir, 'Eucnemidae_Microrhagus_lepidus')  
validation_eucnemidae_microrhagus_pygmaeus_dir = os.path.join(validation_dir, 'Eucnemidae_Microrhagus_pygmaeus')
validation_salpingidae_rabocerus_foveolatus_dir = os.path.join(validation_dir, 'Salpingidae_Rabocerus_foveolatus')  
validation_salpingidae_salpingus_palnirostris_dir = os.path.join(validation_dir, 'Salpingidae_Salpingus_palnirostris')

num_eucnemidae_hylis_foveicollis_tr = len(os.listdir(train_eucnemidae_hylis_foveicollis_dir))
num_eucnemidae_hylis_olexai_tr = len(os.listdir(train_eucnemidae_hylis_olexai_dir))
num_eucnemidae_microrhagus_lepidus_tr = len(os.listdir(train_eucnemidae_microrhagus_lepidus_dir))
num_eucnemidae_microrhagus_pygmaeus_tr = len(os.listdir(train_eucnemidae_microrhagus_pygmaeus_dir))
num_salpingidae_rabocerus_foveolatus_tr = len(os.listdir(train_salpingidae_rabocerus_foveolatus_dir))
num_salpingidae_salpingus_palnirostris_tr = len(os.listdir(train_salpingidae_salpingus_palnirostris_dir))

num_eucnemidae_hylis_foveicollis_val = len(os.listdir(validation_eucnemidae_hylis_foveicollis_dir))
num_eucnemidae_hylis_olexai_val = len(os.listdir(validation_eucnemidae_hylis_olexai_dir))
num_eucnemidae_microrhagus_lepidus_val = len(os.listdir(validation_eucnemidae_microrhagus_lepidus_dir))
num_eucnemidae_microrhagus_pygmaeus_val = len(os.listdir(validation_eucnemidae_microrhagus_pygmaeus_dir))
num_salpingidae_rabocerus_foveolatus_val = len(os.listdir(validation_salpingidae_rabocerus_foveolatus_dir))
num_salpingidae_salpingus_palnirostris_val = len(os.listdir(validation_salpingidae_salpingus_palnirostris_dir))

total_train = num_eucnemidae_hylis_foveicollis_tr + num_eucnemidae_hylis_olexai_tr + num_eucnemidae_microrhagus_lepidus_tr + num_eucnemidae_microrhagus_pygmaeus_tr + num_salpingidae_salpingus_palnirostris_tr + num_salpingidae_rabocerus_foveolatus_tr
total_val = num_eucnemidae_hylis_foveicollis_val + num_eucnemidae_hylis_olexai_val + num_eucnemidae_microrhagus_lepidus_val + num_eucnemidae_microrhagus_pygmaeus_val + num_salpingidae_salpingus_palnirostris_val + num_salpingidae_rabocerus_foveolatus_val

print('total training eucnemidae hylis foveicollis images:', num_eucnemidae_hylis_foveicollis_tr)
print('total training eucnemidae hylis olexai images:', num_eucnemidae_hylis_olexai_tr)
print('total training eucnemidae microrhagus lepidus images:', num_eucnemidae_microrhagus_lepidus_tr)
print('total training eucnemidae microrhagus pygmaeus images:', num_eucnemidae_microrhagus_pygmaeus_tr)
print('total training salpingidae rabocerus foveolatus images:', num_salpingidae_rabocerus_foveolatus_tr)
print('total training Salpingidae images:', num_salpingidae_salpingus_palnirostris_tr)

print('total validation eucnemidae hylis foveicollis images:', num_eucnemidae_hylis_foveicollis_val)
print('total validation eucnemidae hylis olexai images:', num_eucnemidae_hylis_olexai_val)
print('total validation eucnemidae microrhagus lepidus images:', num_eucnemidae_microrhagus_lepidus_val)
print('total validation eucnemidae microrhagus pygmaeus images:', num_eucnemidae_microrhagus_pygmaeus_val)
print('total validation salpingidae rabocerus foveolatus images:', num_salpingidae_rabocerus_foveolatus_val)
print('total validation salpingidae salpingus palnirostris images:', num_salpingidae_salpingus_palnirostris_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='multi_output')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='multi_output')

sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary() 

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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

show_graph(tf.get_default_graph().as_graph_def())
