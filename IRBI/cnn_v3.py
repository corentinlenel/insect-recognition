from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

path_to_db = '/home/corentin/Documents/IRBI'
PATH = os.path.join(path_to_db)

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_abeille_dir = os.path.join(train_dir, 'Abeille')  
train_acalyptrate_dir = os.path.join(train_dir, 'Acalyptraté')
train_araignee_dir = os.path.join(train_dir, 'Araignée')
train_bourdon_dir = os.path.join(train_dir, 'Bourdon')
train_carabe_dir = os.path.join(train_dir, 'Carabe')
train_caraboide_dir = os.path.join(train_dir, 'Caraboide')
train_charancon_dir = os.path.join(train_dir, 'Charançon')
train_chilopode_dir = os.path.join(train_dir, 'Chilopode')
train_cicadelle_dir = os.path.join(train_dir, 'Cicadelle')
train_cloporte_dir = os.path.join(train_dir, 'Cloporte')
train_coccinelle_dir = os.path.join(train_dir, 'Coccinelle')
train_collembole_dir = os.path.join(train_dir, 'Collembole')
train_cecidomyie_dir = os.path.join(train_dir, 'Cécidomyie')
train_diplopode_dir = os.path.join(train_dir, 'Diplopode')
train_dolichopodide_dir = os.path.join(train_dir, 'Dolichopodide')
train_fourmis_dir = os.path.join(train_dir, 'Fourmis Noire')
train_guepe_dir = os.path.join(train_dir, 'Guèpe')
train_larve_dir = os.path.join(train_dir, 'Larve Coléoptère')
train_mouche_dir = os.path.join(train_dir, 'Mouche')
train_moustique_dir = os.path.join(train_dir, 'Moustique')
train_myrmica_dir = os.path.join(train_dir, 'Myrmica')
train_papillon_dir = os.path.join(train_dir, 'Papillon')
train_parasitoide_dir = os.path.join(train_dir, 'Parasitoide')
train_puceron_dir = os.path.join(train_dir, 'Puceron')
train_punaise_dir = os.path.join(train_dir, 'Punaise')
train_sauterelle_dir = os.path.join(train_dir, 'Sauterelle')
train_scaraboide_dir = os.path.join(train_dir, 'Scaraboide')
train_staphylin_dir = os.path.join(train_dir, 'Staphylin')
train_syrphe_dir = os.path.join(train_dir, 'Syrphe')
train_tipule_dir = os.path.join(train_dir, 'Tipule')

validation_abeille_dir = os.path.join(validation_dir, 'Abeille')  
validation_acalyptrate_dir = os.path.join(validation_dir, 'Acalyptraté')
validation_araignee_dir = os.path.join(validation_dir, 'Araignée')
validation_bourdon_dir = os.path.join(validation_dir, 'Bourdon')
validation_carabe_dir = os.path.join(validation_dir, 'Carabe')
validation_caraboide_dir = os.path.join(validation_dir, 'Caraboide')
validation_charancon_dir = os.path.join(validation_dir, 'Charançon')
validation_chilopode_dir = os.path.join(validation_dir, 'Chilopode')
validation_cicadelle_dir = os.path.join(validation_dir, 'Cicadelle')
validation_cloporte_dir = os.path.join(validation_dir, 'Cloporte')
validation_coccinelle_dir = os.path.join(validation_dir, 'Coccinelle')
validation_collembole_dir = os.path.join(validation_dir, 'Collembole')
validation_cecidomyie_dir = os.path.join(validation_dir, 'Cécidomyie')
validation_diplopode_dir = os.path.join(validation_dir, 'Diplopode')
validation_dolichopodide_dir = os.path.join(validation_dir, 'Dolichopodide')
validation_fourmis_dir = os.path.join(validation_dir, 'Fourmis Noire')
validation_guepe_dir = os.path.join(validation_dir, 'Guèpe')
validation_larve_dir = os.path.join(validation_dir, 'Larve Coléoptère')
validation_mouche_dir = os.path.join(validation_dir, 'Mouche')
validation_moustique_dir = os.path.join(validation_dir, 'Moustique')
validation_myrmica_dir = os.path.join(validation_dir, 'Myrmica')
validation_papillon_dir = os.path.join(validation_dir, 'Papillon')
validation_parasitoide_dir = os.path.join(validation_dir, 'Parasitoide')
validation_puceron_dir = os.path.join(validation_dir, 'Puceron')
validation_punaise_dir = os.path.join(validation_dir, 'Punaise')
validation_sauterelle_dir = os.path.join(validation_dir, 'Sauterelle')
validation_scaraboide_dir = os.path.join(validation_dir, 'Scaraboide')
validation_staphylin_dir = os.path.join(validation_dir, 'Staphylin')
validation_syrphe_dir = os.path.join(validation_dir, 'Syrphe')
validation_tipule_dir = os.path.join(validation_dir, 'Tipule')

num_abeille_tr = len(os.listdir(train_abeille_dir))
num_acalyptrate_tr = len(os.listdir(train_acalyptrate_dir))
num_araignee_tr = len(os.listdir(train_araignee_dir))
num_bourdon_tr = len(os.listdir(train_bourdon_dir))
num_carabe_tr = len(os.listdir(train_carabe_dir))
num_caraboide_tr = len(os.listdir(train_caraboide_dir))
num_charancon_tr = len(os.listdir(train_charancon_dir))
num_chilopode_tr = len(os.listdir(train_chilopode_dir))
num_cicadelle_tr = len(os.listdir(train_cicadelle_dir))
num_cloporte_tr = len(os.listdir(train_cloporte_dir))
num_coccinelle_tr = len(os.listdir(train_coccinelle_dir))
num_collembole_tr = len(os.listdir(train_collembole_dir))
num_cecidomyie_tr = len(os.listdir(train_cecidomyie_dir))
num_diplopode_tr = len(os.listdir(train_diplopode_dir))
num_dolichopodide_tr = len(os.listdir(train_dolichopodide_dir))
num_fourmis_tr = len(os.listdir(train_fourmis_dir))
num_guepe_tr = len(os.listdir(train_guepe_dir))
num_larve_tr = len(os.listdir(train_larve_dir))
num_mouche_tr = len(os.listdir(train_mouche_dir))
num_moustique_tr = len(os.listdir(train_moustique_dir))
num_myrmica_tr = len(os.listdir(train_myrmica_dir))
num_papillon_tr = len(os.listdir(train_papillon_dir))
num_parasitoide_tr = len(os.listdir(train_parasitoide_dir))
num_puceron_tr = len(os.listdir(train_puceron_dir))
num_punaise_tr = len(os.listdir(train_punaise_dir))
num_sauterelle_tr = len(os.listdir(train_sauterelle_dir))
num_scaraboide_tr = len(os.listdir(train_scaraboide_dir))
num_staphylin_tr = len(os.listdir(train_staphylin_dir))
num_syrphe_tr = len(os.listdir(train_syrphe_dir))
num_tipule_tr = len(os.listdir(train_tipule_dir))

num_abeille_val = len(os.listdir(validation_Abeille_dir))
num_acalyptrate_val = len(os.listdir(validation_acalyptrate_dir))
num_araignee_val = len(os.listdir(validation_araignee_dir))
num_bourdon_val = len(os.listdir(validation_bourdon_dir))
num_carabe_val = len(os.listdir(validation_carabe_dir))
num_caraboide_val = len(os.listdir(validation_caraboide_dir))
num_charancon_val = len(os.listdir(validation_charancon_dir))
num_chilopode_val = len(os.listdir(validation_chilopode_dir))
num_cicadelle_val = len(os.listdir(validation_cicadelle_dir))
num_cloporte_val = len(os.listdir(validation_cloporte_dir))
num_coccinelle_val = len(os.listdir(validation_coccinelle_dir))
num_collembole_val = len(os.listdir(validation_collembole_dir))
num_cecidomyie_val = len(os.listdir(validation_cecidomyie_dir))
num_diplopode_val = len(os.listdir(validation_diplopode_dir))
num_dolichopodide_val = len(os.listdir(validation_dolichopodide_dir))
num_fourmis_val = len(os.listdir(validation_fourmis_dir))
num_guepe_val = len(os.listdir(validation_guepe_dir))
num_larve_val = len(os.listdir(validation_larve_dir))
num_mouche_val = len(os.listdir(validation_mouche_dir))
num_moustique_val = len(os.listdir(validation_moustique_dir))
num_myrmica_val = len(os.listdir(validation_myrmica_dir))
num_papillon_val = len(os.listdir(validation_papillon_dir))
num_parasitoide_val = len(os.listdir(validation_parasitoide_dir))
num_puceron_val = len(os.listdir(validation_puceron_dir))
num_punaise_val = len(os.listdir(validation_punaise_dir))
num_sauterelle_val = len(os.listdir(validation_sauterelle_dir))
num_scaraboide_val = len(os.listdir(validation_scaraboide_dir))
num_staphylin_val = len(os.listdir(validation_staphylin_dir))
num_syrphe_val = len(os.listdir(validation_syrphe_dir))
num_tipule_val = len(os.listdir(validation_tipule_dir))

total_train = num_abeille_tr + num_acalyptrate_tr + num_araignee_tr + num_bourdon_tr + num_carabe_tr + num_caraboide_tr + num_charancon_tr + num_chilopode_tr + num_cicadelle_tr + num_cloporte_tr + num_coccinelle_tr + num_collembole_tr + num_cecidomyie_tr + num_diplopode_tr + num_dolichopodide_tr + num_fourmis_tr + num_guepe_tr + num_larve_tr + num_mouche_tr + num_moustique_tr + num_myrmica_tr + num_papillon_tr + num_parasitoide_tr + num_puceron_tr + num_punaise_tr + num_sauterelle_tr + num_scaraboide_tr + num_staphylin_tr + num_syrphe_tr + num_tipule_tr
total_val = num_abeille_val + num_acalyptrate_val + num_araignee_val + num_bourdon_val + num_carabe_val + num_caraboide_val + num_charancon_val + num_chilopode_val + num_cicadelle_val + num_cloporte_val + num_coccinelle_val + num_collembole_val + num_cecidomyie_val + num_diplopode_val + num_dolichopodide_val + num_fourmis_val + num_guepe_val + num_larve_val + num_mouche_val + num_moustique_val + num_myrmica_val + num_papillon_val + num_parasitoide_val + num_puceron_val + num_punaise_val + num_sauterelle_val + num_scaraboide_val + num_staphylin_val + num_syrphe_val + num_tipule_val

print('total training abeille images:', num_abeille_tr)
print('total training acalyptrate images:', num_acalyptrate_tr)
print('total training araignee images:', num_araignee_tr)
print('total training bourdon images:', num_bourdon_tr)
print('total training carabe images:', num_carabe_tr)
print('total training caraboide images:', num_caraboide_tr)
print('total training charancon images:', num_charancon_tr)
print('total training chilopode images:', num_chilopode_tr)
print('total training cicadelle images:', num_cicadelle_tr)
print('total training cloporte images:', num_cloporte_tr)
print('total training coccinelle images:', num_coccinelle_tr)
print('total training collembole images:', num_collembole_tr)
print('total training cecidomyie images:', num_cecidomyie_tr)
print('total training diplopode images:', num_diplopode_tr)
print('total training dolichopodide images:', num_dolichopodide_tr)
print('total training fourmis images:', num_fourmis_tr)
print('total training guepe images:', num_guepe_tr)
print('total training larve images:', num_larve_tr)
print('total training mouche images:', num_mouche_tr)
print('total training moustique images:', num_moustique_tr)
print('total training myrmica images:', num_myrmica_tr)
print('total training papillon images:', num_papillon_tr)
print('total training parasitoide images:', num_parasitoide_tr)
print('total training puceron images:', num_puceron_tr)
print('total training punaise images:', num_punaise_tr)
print('total training sauterelle images:', num_sauterelle_tr)
print('total training scaraboide images:', num_scaraboide_tr)
print('total training staphylin images:', num_staphylin_tr)
print('total training syrphe images:', num_syrphe_tr)
print('total training tipule images:', num_tipule_tr)

print('total validation abeille images:', num_abeille_val)
print('total validation acalyptrate images:', num_acalyptrate_val)
print('total validation araignee images:', num_araignee_val)
print('total validation bourdon images:', num_bourdon_val)
print('total validation carabe images:', num_carabe_val)
print('total validation caraboide images:', num_caraboide_val)
print('total validation charancon images:', num_charancon_val)
print('total validation chilopode images:', num_chilopode_val)
print('total validation cicadelle images:', num_cicadelle_val)
print('total validation cloporte images:', num_cloporte_val)
print('total validation coccinelle images:', num_coccinelle_val)
print('total validation collembole images:', num_collembole_val)
print('total validation cecidomyie images:', num_cecidomyie_val)
print('total validation diplopode images:', num_diplopode_val)
print('total validation dolichopodide images:', num_dolichopodide_val)
print('total validation fourmis images:', num_fourmis_val)
print('total validation guepe images:', num_guepe_val)
print('total validation larve images:', num_larve_val)
print('total validation mouche images:', num_mouche_val)
print('total validation moustique images:', num_moustique_val)
print('total validation myrmica images:', num_myrmica_val)
print('total validation papillon images:', num_papillon_val)
print('total validation parasitoide images:', num_parasitoide_val)
print('total validation puceron images:', num_puceron_val)
print('total validation punaise images:', num_punaise_val)
print('total validation sauterelle images:', num_sauterelle_val)
print('total validation scaraboide images:', num_scaraboide_val)
print('total validation staphylin images:', num_staphylin_val)
print('total validation syrphe images:', num_syrphe_val)
print('total validation tipule images:', num_tipule_val)
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
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

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
