from gc import callbacks
from PIL import Image
from glob import glob, iglob
import os, sys
import numpy as np

from tensorflow import keras 
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer

from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

normalize = Normalizer()

path_train = "./images"
path_pred = "./predict"

size_x = 200
size_y = 200

def load_classes(path):
    classes = next(os.walk(path))
    classes=classes[1]
    return classes

def load_dataset(path):
    data_images = []
    data_classes = []
    
    classes = load_classes(path)
        
    for classe in classes:
        im_path = str(path + "/" + classe + '/*')
        for files in iglob(im_path, recursive=True):
            image = Image.open(files)

            image = image.resize((size_x, size_y))           
                        
            data_images.append(np.asarray(image))
            data_classes.append(classe)
            
            image = image.rotate(-90)
            # # show the image          
    
    images = np.array(data_images)
        
    return images, data_classes, classes

    
def image_generator():
    
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)   
    
    classes = load_classes()
    
    for classe in classes:
        im_path = str(path_train + "/" + classe + '/*')
        for files in iglob(im_path, recursive=True):
            print(files)
            image = Image.open(files)                
            image = np.array(image)  # this is a Numpy array with shape (3, 150, 150)
            image = image.reshape((1,) + image.shape) 
                        
            i = 0
            save_path = str(path_train + '/' + classe + '/')
            for batch in datagen.flow(image, batch_size=1,
                                    save_to_dir=save_path, save_prefix=classe, save_format='png'):
                i += 1
                if i > 20:
                    break  
    
    sys.exit()   
        
def pre_processing(images, labels, classes):    
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
                        
    X_train, X_test, y_train, y_test = train_test_split(images, integer_encoded, test_size=0.33, random_state=42)
             
    X_train=X_train/255.0
    X_test=X_test/255.0
    
    X_train = X_train.reshape(len(X_train),size_x, size_y, 3)
    X_test = X_test.reshape(len(X_test),size_x, size_y, 3)
    
    return X_train, X_test, y_train, y_test


def cnn_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=(size_x, size_y, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), input_shape=(size_x, size_y, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), input_shape=(size_x, size_y, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
   
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    model.summary()
    
    return model


def split_data():

    images, labels, classes = load_dataset(path_train)
    X_train, X_test, y_train, y_test = pre_processing(images, labels, classes)

    return X_train, X_test, y_train, y_test

    
def train_cnn(model, X_train, y_train):

    # es_cb = EarlyStopping(monitor='loss')
   
    model.fit(X_train, y_train, epochs=10) #, callbacks=[es_cb])

    model.save("final_model.h5")


def test_cnn(X_test, y_test):

    model = load_model("final_model.h5")

    score = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=-1)

    y_test = y_test.reshape(-1)

    print(confusion_matrix(y_test, y_pred))

    print(accuracy_score(y_test, y_pred) * 100)


def pre_processing_predict(images, labels):    
    
    label_encoder = LabelEncoder()
    y_pred = label_encoder.fit_transform(labels)               
             
    X_pred = images/255.0
     
    X_pred = X_pred.reshape(len(X_pred),size_x, size_y, 3)
    
    return X_pred, y_pred

def predict_cnn():

    images, labels, classes = load_dataset(path_pred)

    X_pred, y_pred = pre_processing_predict(images, labels)

    model = load_model('final_model.h5')

    print(model.predict(X_pred))

    pred = np.argmax(model.predict(X_pred), axis=-1)

    print(confusion_matrix(y_pred, pred))

    print(accuracy_score(y_pred, pred) * 100)


# image_generator()  

X_train, X_test, y_train, y_test = split_data()

model = cnn_model()

train_cnn(model, X_train, y_train)

test_cnn(X_test, y_test)

predict_cnn()