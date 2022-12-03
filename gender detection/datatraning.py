from __future__ import absolute_import, division, print_function, unicode_literals
import csv
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf1


train_img = []
train_labels = []
test_img = []
test_labels = []
class_names = ['female','male']

def getPathImg():
    with open('pathimg.csv') as pathImg:
        path = csv.reader(pathImg, delimiter=',')
        for row in path:
            return row

def getGenderImg():
    with open('testdata.csv') as testData :
        genders = csv.reader(testData, delimiter=',')
        for row in genders:
           return row

def setTrainingData():
    pathImg = getPathImg()
    genders = getGenderImg()
    i = 0
    for index,img in enumerate(pathImg):
        if genders[index] != "NaN":
            path = os.path.join("wiki_crop",img)
            if i < 47749:
                os.rename(path,"Dataset/train-data/"+class_names[int(genders[index])]+"/"+str(i)+".jpg")
            else :
                os.rename(path,"Dataset/test-data/"+class_names[int(genders[index])]+"/"+str(i)+".jpg")
            i=i+1
            
def buildModel():
    train_dir = "Dataset/train-data"
    test_dir = "Dataset/test-data"
    train_male_dir = os.path.join("Dataset/train-data", 'male')  
    train_female_dir = os.path.join("Dataset/train-data", 'female')
    test_male_dir = os.path.join("Dataset/test-data", 'male')  
    test_female_dir = os.path.join("Dataset/test-data", 'female')  
    total_train = len(os.listdir(train_male_dir))+len(os.listdir(train_female_dir))
    total_test = len(os.listdir(test_male_dir))+len(os.listdir(test_female_dir))

    batch_size = 128    
    epochs = 4
    IMG_HEIGHT = 257
    IMG_WIDTH = 257

    train_image_generator = ImageDataGenerator(rescale=1./255) 
    test_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=train_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
    
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=test_data_gen,
        validation_steps=total_test // batch_size
    )
    model.save('genderDetection-model.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

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

                                       

def testModel():
    new_model = tf.keras.models.load_model('genderDetection-model.h5')
    new_model.summary()
    img = image.load_img('testimages/male4.jpg',target_size=(257,257))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)

    result = new_model.predict(img)
    print(result[0][0])
    if result[0][0] <= 0.5:
        print("female")
    else :
        print("male")

testModel()







