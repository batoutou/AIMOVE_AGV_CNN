from PIL import Image
from glob import glob, iglob
import os, sys
import numpy as np

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

path = "./images"

def load_classes():
    classes = next(os.walk(path))
    classes=classes[1]
    return classes

def load_dataset():
    data_images = []
    data_classes = []
    
    classes = load_classes()
        
    for classe in classes:
        im_path = str(path + '/' + classe + '/*')
        for files in iglob(im_path, recursive=True):
            image = Image.open(files)
            # image = np.array(image)
            # data = [np.resize(image, (400,400,4)), classe]            
            # array = np.array(data)
            
            data_images.append(np.asarray(image))
            data_classes.append(classe)
            
            # # show the image
            # image.show()   
            
    
    images = np.array(data_images)
        
    return images, data_classes, classes

    
def image_generator():
    
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)   
    
    classes = load_classes()
    
    for classe in classes:
        im_path = str(path + '/' + classe + '/*')
        for files in iglob(im_path, recursive=True):
            image = Image.open(files)                
            image = np.array(image)  # this is a Numpy array with shape (3, 150, 150)
            image = image.reshape((1,) + image.shape) 
                        
            i = 0
            save_path = str(path + '/' + classe + '/')
            for batch in datagen.flow(image, batch_size=1,
                                    save_to_dir=save_path, save_prefix=classe, save_format='png'):
                i += 1
                if i > 20:
                    break  
    
    sys.exit()   
        
def pre_processing(images, labels, classes):    
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_data = onehot_encoder.fit_transform(integer_encoded)
                
    X_train, X_test, y_train, y_test = train_test_split(images, y_data, test_size=0.33, random_state=42)
             
    # X_train=X_train/255.0
    # X_test=X_test/255.0
    
    X_train = X_train.reshape(len(X_train),667, 604, 3)
    X_test = X_test.reshape(len(X_test),667, 604, 3)
    
    return X_train, X_test, y_train, y_test


def cnn_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(667, 604, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    model.summary()
    
    return model

    
def train_cnn():
    # image_generator()    
    
    images, labels, classes = load_dataset()
    X_train, X_test, y_train, y_test = pre_processing(images, labels, classes)
    model = cnn_model()
   
    model.fit(X_train, y_train, epochs=10)
    
    
    
train_cnn()

