from __future__ import absolute_import, division, print_function, unicode_literals

import glob

import cv2
# Helper libraries
import numpy as np
#import pandas as pd
from keras.constraints import max_norm
# TensorFlow and tf.keras
from tensorflow import keras

import matplotlib.pyplot as plt


def getAge(name):
    begin = name.split('\\')[2]
    return int(begin.split('_')[0])

def getAgeClass(age):
    if age <= 25:
        return 0
    elif 25 < age < 36:
        return 1
    elif 35 < age < 46:
        return 2
    elif 45 < age < 61:
        return 3
    else:
        return 4

train_images = []
test_images = []
train_labels = []
test_labels = []
ages = []

nb_u20 = 0
nb_u40 = 0
nb_u60 = 0
nb_u80 = 0
nb_u100 = 0

# takes 250 images in every class
for file in glob.glob('Adience\\train_4600\\*.jpg'):
    pic = cv2.imread(file)
    pic = pic / 255
    age = getAge(file)
    max_img = 250
    if 0 < age < 100:
        if 0 < age < 20 and nb_u20 < max_img:
            nb_u20 = nb_u20 + 1
            train_images.append(pic)
            train_labels.append(age // 20)
        elif 19 < age < 40 and nb_u40 < max_img:
            nb_u40 = nb_u40 + 1
            train_images.append(pic)
            train_labels.append(age // 20)
        elif 39 < age < 60 and nb_u60 < max_img:
            nb_u60 = nb_u60 + 1
            train_images.append(pic)
            train_labels.append(age // 20)
        elif 59 < age < 80 and nb_u80 < max_img:
            nb_u80 = nb_u80 + 1
            train_images.append(pic)
            train_labels.append(age // 20)
        elif 79 < age < 100 and nb_u100 < max_img:
            nb_u100 = nb_u100 + 1
            train_images.append(pic)
            train_labels.append(age // 20)

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

for file in glob.glob('Adience\\test\\*.jpg'):
    pic = cv2.imread(file)
    pic = pic / 255
    age = getAge(file)
    if 0 < age < 100:
        test_images.append(pic)
        test_labels.append(age//20)
        ages.append(age)

test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

class_names = [
    '0-19',
    '20-39',
    '40-59',
    '60-79',
    '80-99']

model = keras.Sequential()


#Modèle complexe

model.add(keras.layers.Conv2D(32, (3, 3), kernel_constraint=max_norm(3), bias_constraint=max_norm(3),
                              input_shape=(200, 200, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(64, kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='relu'))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer='abdennour', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, validation_split=0.2, batch_size=32, shuffle=True)


#Modèle simplifié

model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer='abdennour', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, validation_split=0.2, batch_size=32, shuffle=True)

model.save('age_recognition.h5')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)
np.argmax(predictions[0])


# affiche les probabilités de chaque classe pour les images de test
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

model.load_weights('age_recognition.h5')
predictions = model.predict(test_images)

pred_df = pd.DataFrame(predictions)
pred_df.columns = class_names

pred_df.insert(0, 'real', ages)

print(pred_df)


