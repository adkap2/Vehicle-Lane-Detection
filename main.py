import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import test as tests
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Softmax, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# here definition of constants, etc.
#IMAGE_SHAPE = (160, 576)
NUM_CLASSES = 2

def get_data():

    pkl = open('full_CNN_labels.p', 'rb')
    labels = pickle.load(pkl)
    pkl = open('full_CNN_train.p', 'rb')
    imgs = pickle.load(pkl)

    return imgs, labels

def build_model(input_shape):
    """ Baseline model"""
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    #model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    model.add(Dropout(0.2))

    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    model.add(Dropout(0.2))

    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    model.add(Dropout(0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    model.add(Dropout(0.2))

    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
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

# def optimize(model, label, learning_rate, num_classes):

#     with tf.name_scope("logits"):
#         logits_ = tf.reshape(model, (-1, num_classes))
#     labels_ = tf.reshaoe(label, (-1, num_classes))
#     with tf.name_scope("softmax"):
#         soft_max = tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=labels_)
#     with tf.name_scope("xent"):
#         cross_entropy_loss = tf.reduce_mean(soft_max)
#     with tf.name_scope("train"):
#         optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cross_entropy_loss)

#     tf.summary.histogram("logits", logits_)
#     tf.summary.histogram("softmax", soft_max)
#     tf.summary.scalar("xent", cross_entropy_loss)
    
#     return (logits_, optimizer, cross_entropy_loss)

def fit_model(imgs, labels, model):

    train_images = np.array(imgs)
    labels = np.array(labels)

    labels = labels / 255
    train_images, labels = shuffle(train_images, labels)

    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)
    

    batch_size = 128
    epochs = 10

    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)



    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
        epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    model.save('full_CNN_model.h5')
    model.summary()
    print(model.evaluate(X_train, y_train))

    return history


# def run():
#     data_dir = './data'
#     runs_dir = './runs'
#     save_file = './save/fcn_model_ckpt_' + time.strftime("%Y%m%d_%H%M%S")
#     tests.test_for_kitti_dataset(data_dir)

#     # helper.generate_augmented_images(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

#     # get_batches_fn = helper.get_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

#     #correct_label = tf.Variable(tf.float32, shape = [None, None, None, NUM_CLASSES])
#     #learning_rate = tf.Variable(tf.float32)

#     model = build_model()
#     print(model.summary())

#     #X_train, y_train = helper.get_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

#     #history = model.fit(X_train, y_train)
#     model.fit_generator(generator=helper.get_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE))
#     #logits, train_op, cross_entropy_loss = optimize(model, correct_label, learning_rate, NUM_CLASSES)


def main():

    imgs, labels = get_data()
    input_shape = np.array(imgs).shape[1:]

    model = build_model(input_shape)
    history = fit_model(imgs, labels, model)

if __name__ == '__main__':
    # run()
    main()