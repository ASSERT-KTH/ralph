## Return 1 if the input binary is detected as a non-malware
from pandarize import preprocess_csv
from utils import *
import sys
import pandarize
import argparse
import pandas as pd
import os
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_keras
from concurrent.futures import ThreadPoolExecutor

import keras

import matplotlib.pyplot as plt
import os

from datetime import datetime

import tensorflow




print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))


class MINOS:
    '''
        A class for the implementation of the MINOS proposal (https://www.ndss-symposium.org/wp-content/uploads/ndss2021_4C-4_24444_paper.pdf)
        ...

        Attributes
        ----------

        classes: list
            Do not change this attribute, it contains the mapping from the literal CLASS value: BENIGN or MALIGN as indexes in the array
            Use this attribute to refer to the index of the correct label,
            for example, `MINOS.classes.index("MALIGN")` or `MINOS.classes[0]`
    '''
    classes = ['MALIGN', 'BENIGN']

    def __init__(self, size=(100,100), add_maxpool=True):
        """
        Parameters
        ----------
        size : tuple
            The size of the image representation, (100,100) by default
        add_maxpool : boolean
            The option to add maxpool layers between convolutional layers
        """
        # Create and compile the CNN following the instructions from MINOS
        model = Sequential()
        model.add(Reshape((*size, 1), input_shape=(size[0]*size[1], )))
        model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(*size,1)))
        if add_maxpool:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        if add_maxpool:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        if add_maxpool:
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        # To print the summary of the CNN model, layers, parameters, etc
        model.summary()

        self.model = model


    #
    #
    def preprocess(self, data, shape=(100,100)):
        """
        This method returns a collection of image pixels.

        Each pixel value is stored in a column with the name `<row>_<column>`
        for the 100x100 binary transformation. There should be a total of 100x100 columns then. To get
        the column values you can construct the following array `[f"{x}_{y}" for x in range(shape[0]) for y in range(shape[1])]`
        and then access the pandas frame.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset frame
        shape: tuple
            Image tuple size (width, height), default value as (100,100)
        """

        features = [f"{x}_{y}" for x in range(shape[0]) for y in range(shape[1])]
        # Categorize data
        labels = data['CLASS'].apply(lambda x: MINOS.classes.index(x))
        labels = to_categorical(labels)

        values = data[features].values
        return values, np.array(labels)

    def load(self, model_name="minos.h5"):
        self.model = keras.models.load_model(model_name)

    def fit(self, train_data, test_data, epochs=50, model_name="minos.h5", test = True):

        """
        Trains the classifier


        Parameters
        ----------
        train_data : pandas.DataFrame
            The dataset frame used for training
        test_data: pandas.DataFrame
            The dataset frame used for validation and testing during training
        epochs:
            Number of epochs of training, 50 epochs is the default
        mode_name: str
            The model can be saved as a h5 file after training, the model will be saved with this parameter value as name
            The training will be avoided if there is a file with this name.
        """

        # This check, will avoid to train the model again if the model
        # was already saved in the filesystem. This will help us to prevent
        # the model creating on every exeution of the fit method.
        # Some counting for info
        X_train, Y_train = self.preprocess(train_data)


        if test:
            X_test, Y_test = self.preprocess(test_data)
            history = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), callbacks=[], epochs=epochs)
        else:
            # otherwise use all the data for training
            history = self.model.fit(X_train, Y_train, callbacks=[], epochs=epochs)

        self.model.save(f"{model_name}.h5")

        if model_name.endswith(".tf"):
            import tensorflow as tf
            tf.saved_model.save(self.model, model_name)



    def predict(self, data):
        """
        Given dataframe, uses the fitted model to
        predict the labels

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset frame containing the instances to be predicted
        """
        X, _ = self.preprocess(data)

        p = self.model.predict(X)

        d = pd.DataFrame(p, columns=MINOS.classes)
        return d


    def predict_classes(self, pd,  predictions):
        """
         The predictions (predict method) are given using a column per label,
         and setting the row values to the probability of the instance to be that
         class. This method, adds a new column, 'CLASS', and sets its value to the
         name of the prediction column with higher probability.

        Parameters
        ----------
        pd : pandas.DataFrame
            The dataset frame where the `CLASS` label will be added
        predictions: pandas.DataFrame
            Outcome of the predict method
        """
        cp = pd.reset_index()
        cp['CLASS'] = predictions.idxmax(axis=1)
        return cp


def train(args):
    datasetb = pd.read_csv(args.benign)
    datasetm = pd.read_csv(args.malign)

    # balance the classes
    minlen = min(len(datasetb), len(datasetm))
    datasetb = datasetb.sample(minlen)
    datasetm = datasetm.sample(minlen)

    dataset = pd.concat([datasetb, datasetm], axis=0)
    print(len(datasetb), len(datasetm))
    ACC = 0
    TOTAL = 0
    for index, row in dataset.iterrows():

        print(f"\r{row.Name} for testing                          ")
    # Split the dataset into train and test
        train = dataset.drop(index=index)
        real_test = pd.DataFrame([row])

        # train, test = split(train, testfraction=1.0)
        # print(train)

        #print(test)

        t = time.time()
        minos = MINOS()
        # Do not use data to test, it is to few
        # Instead we are using a single instance to test
        minos.fit(train, None, model_name=args.model, epochs=50, test=False)
        t = time.time()
        print("Time training", time.time() - t)
        pr = minos.predict(real_test)
        print(pr)
        # This is veeery conservative
        if pr[row.CLASS][0] > 0.9:
            ACC += 1

        TOTAL += 1

        sys.stderr.write(f"\rAccuracy {pr[row.CLASS][0]} {ACC}/{TOTAL} = {ACC/TOTAL*100:.2f}%      ")

            #



def normal_train(args, unify_classes=False):
    datasetb = pd.read_csv(args.benign)
    datasetm = pd.read_csv(args.malign)
    # balance the classes
    if unify_classes:
        minlen = min(len(datasetb), len(datasetm))
        datasetb = datasetb.sample(minlen)
        datasetm = datasetm.sample(minlen)


    dataset = pd.concat([datasetb, datasetm], axis=0)

    print(len(datasetb), len(datasetm))
    train, test = split(dataset, 0.8)
    t = time.time()
    minos = MINOS()

    # Do not use data to test, it is to few
    # Instead we are using a single instance to test
    minos.fit(train, test, model_name=args.model)
    t = time.time()
    print("Time training", time.time() - t)


def predict(args):

    t = time.time()
    minos = MINOS()
    minos.load(model_name=args.model)
    print("Time loading", time.time() - t)
    t = time.time()
    pixels = preprocess_csv(args.input)
    print("Time preprocess", time.time() - t)
    predictions = minos.predict(pixels)
    print(predictions)
    exit(int(predictions['BENIGN'].values[-1]))


if __name__ == '__main__':

    CURRENT_DIR = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='MINOS')
    subparser = parser.add_subparsers(help='Subcommands')
    predict_parser = subparser.add_parser('predict')
    predict_parser.add_argument('--model', type=str, help='Model file', default=f'{CURRENT_DIR}/original23.h5')
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument('-i', '--input', type=str, help='Input dataset', required=True)

    train_parser = subparser.add_parser('train-one-off', help="Train MINOS with a one-off strategy for validation")
    train_parser.add_argument('-b', '--benign', type=str, help='Benign dataset', required=True)
    train_parser.add_argument('-m', '--malign', type=str, help='Malign dataset', required=True)

    train_parser.add_argument('--model', type=str, help='Model file to save', default=f'{CURRENT_DIR}/original23.h5')
    train_parser.set_defaults(func=train)

    train_parser = subparser.add_parser('train')
    train_parser.add_argument('-b', '--benign', type=str, help='Benign dataset', required=True)
    train_parser.add_argument('-m', '--malign', type=str, help='Malign dataset', required=True)

    train_parser.add_argument('--model', type=str, help='Model file to save', default=f'{CURRENT_DIR}/original23.h5')
    train_parser.set_defaults(func=normal_train)



    args = parser.parse_args()
    args.func(args)


