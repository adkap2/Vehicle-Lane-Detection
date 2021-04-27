import os.path
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Softmax, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.models import load_model

def get_data():
    """ Gets training data and labels to be trained on
    returns raw and labeled images
    """

    pkl = open('full_CNN_labels.p', 'rb')
    labels = pickle.load(pkl)
    pkl = open('full_CNN_train.p', 'rb')
    imgs = pickle.load(pkl)

    return imgs, labels

def save_data(model, dataset):
    """ Saves the generated dataset to a csv file"""
    path = 'dataset.txt'
    df = pd.DataFrame(data=dataset)
    df.to_csv(path)

def build_model(input_shape):
    """ Generates Deep CNN to be used to detect lanes from given driving images
    Does encoding/decoding sequence for images with layers increasing in filter size to a maximum of
    1024 and then decreasing back down to a filter size of 1
    Returns -> compiled model
    """
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    # #model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
    # # Below layers were re-named for easier reading of model summary; this not necessary
    # # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Layer 3
    # model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    model.add(Dropout(0.2))

    # Conv Layer 4
    model.add(Conv2D(256, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    model.add(Dropout(0.2))

    # Conv Layer 5
    model.add(Conv2D(512, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    model.add(Dropout(0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Layer 6
    model.add(Conv2D(512, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    model.add(Dropout(0.2))

    # Conv Layer 7
    model.add(Conv2D(1024, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    model.add(Dropout(0.2))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Upsample 1
    model.add(UpSampling2D(size=(2,2)))

    # Deconv 1
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
    model.add(Dropout(0.2))

    # Deconv 2
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
    model.add(Dropout(0.2))

    # Upsample 2
    model.add(UpSampling2D(size=(2,2)))

    # Deconv 3
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
    model.add(Dropout(0.2))

    # Deconv 4
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
    model.add(Dropout(0.2))

    # Deconv 5
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
    model.add(Dropout(0.2))

    # Upsample 3
    model.add(UpSampling2D(size=(2,2)))

    # Deconv 6
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def fit_model(imgs, labels, model):
    """ Trains compiled model by fitting it to resized training and validation data.
    Trains against provided lane segmentation labels
    Returns -> history (evaluation)"""
    # Converts images to np array
    train_images = np.array(imgs)
    labels = np.array(labels)

    labels = labels / 255 # Normalizes images to readable tensorflow format
    train_images, labels = shuffle(train_images, labels)

    # Traint validation split for images and labels
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

    dataset = [X_train, X_val, y_train, y_val]
    # Arbitrary batch size
    batch_size = 128
    # Currently 10 epochs although model seems to converge early so may
    # Experiement with lower epochs
    epochs = 10
    # Builds data generator to limit amount of image data stored in memory
    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)
    # Sets call backs to stop model training early if loss converges, also saves model after each epoch
    callbacks = [EarlyStopping(monitor='loss', patience=3), ModelCheckpoint(filepath = 'model4.h5', monitor='val_loss', save_best_only=True)]
    # Fits model
    # Test out alternative model
    # model = load_model('baseline_cnn.h5')
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
        epochs=epochs, verbose=1, callbacks=callbacks, validation_data=(X_val, y_val))
    # save_data(model, dataset)
    # model.save('model3.h5')
    model.summary()
    # Evaluates model for loss metrics
    print(model.evaluate(X_train, y_train))

    return history


def main():
    """ Calls all seperate individual functions"""

    imgs, labels = get_data()
    input_shape = np.array(imgs).shape[1:]
    # Builds model with input layers and compiles model using Adam optimizer
    model = build_model(input_shape)
    # Fits model with raw images and trained labels
    history = fit_model(imgs, labels, model)

if __name__ == '__main__':
    main()