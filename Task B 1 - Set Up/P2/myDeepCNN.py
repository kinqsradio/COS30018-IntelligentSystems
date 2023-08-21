import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import math
import json
import sys

import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.models import Model, Sequential
# from keras import initializers
# from tensorflow.keras.layers import Layer, InputSpec
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical


from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import dataset
import argparse

import time
from datetime import timedelta

def build_dataset(data_directory, img_width):
    X, y, tags = dataset.dataset(data_directory, int(img_width))
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = to_categorical(y, nb_classes)
    return feature, label, nb_classes


def build_model(SHAPE, nb_classes, bn_axis, seed=None):
    """
    Build the model
    Args:
      - SHAPE: input image shape
      - nb_classes: number of classes
    Returns:
      - model: built model
    """
    if seed:
        np.random.seed(seed)

    model = Sequential()

    # 1st Convolution Layer
    model.add(Conv2D(32, (3, 3), padding='valid', input_shape=SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution Layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd Convolution Layer
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 1st FC Layer
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 2nd FC Layer
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model

def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='a image dimension', type=int, default=50)
    parser.add_argument('-c', '--channel',
                        help='a image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    # parser.add_argument('-o', '--optimizer',
    #                     help='choose the optimizer (rmsprop, adagrad, adadelta, adam, adamax, nadam)', default="adam")
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="output_result_cnn.txt")
    args = parser.parse_args()
    # dimensions of our images.
    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    SHAPE = (img_width, img_height, channel)
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    data_directory = args.input

    print("loading dataset")
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_directory), args.dimension)
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_directory), args.dimension)
    print("number of classes : {}".format(nb_classes))

    model = build_model(SHAPE, nb_classes, bn_axis)
    # 1.0e-4
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    # Save Model or creates a HDF5 file
    model.save('{}epochs_{}batch_cnn_model_{}.h5'.format(
        epochs, batch_size, data_directory.replace("/", "_")), overwrite=True)
    # del model  # deletes the existing model
    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    tn = cm[0][0]
    if cm.shape[0] > 1:
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
    else:
        # Handle the case where the confusion matrix has only one row.
        # This might mean that one of the classes wasn't predicted at all.
        fn = 0
        tp = 0
        fp = 0
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    TPR = float(tp)/(float(tp)+float(fn))
    FPR = float(fp)/(float(fp)+float(tn))
    accuracy = round((float(tp) + float(tn))/(float(tp) +
                                              float(fp) + float(fn) + float(tn)), 3)
    specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
        (float(tp)+float(fp))
        * (float(tp)+float(fn))
        * (float(tn)+float(fp))
        * (float(tn)+float(fn))
    ), 3)

    f_output = open(args.output, 'a')
    f_output.write('=======\n')
    f_output.write('{}epochs_{}batch_cnn\n'.format(
        epochs, batch_size))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
