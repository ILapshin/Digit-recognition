# MNIST digit recognition neural net model

import numpy as np
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from skimage.transform import rotate, warp, SimilarityTransform
from skimage.util import crop

class Recognizer:

    #=============================================
    # Main class initializes a nueral net model.
    # Train and test data sets should be in the subdirectory .../data by default.
    # As the program is intended especially for Kaggle competition, the sample_submission.csv file must be placed in .../data directory
    #
    # The neural net consists of sequent levels of two convolutional leyers 
    # followed by one maximum pooloing level and two fully connected leyers in the end.
    # 
    # To run recognition properly a fit() method should be applyed after instantiating a Recognazer class object. 
    # A verify() method can be implemented for more visibility.
    #=============================================

    def __init__(self, leyers=(2, 2), train_path='data/train.csv', test_path='data/test.csv'):
        # Leyers is a tuple of number of sequential convolutional levels plus a max pooling level,
        # each item is the number of convolutional levels in a sequence.
        self.leyers = leyers 
        
        self.model = Sequential() # model is a tf.keras.Sequential object thus it can implement all its methods (e.g. summary)               
        for i, item in enumerate(leyers):
            for j in range(item):
                if i == 0 and j == 0:
                    self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
                else:
                    self.model.add(Conv2D(64 * (i + 1), kernel_size=3, activation='relu'))
            self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='sigmoid'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.optimizer.lr=0.01

        # Train and test datasets are read initially while instantiating.
        # They can be rewritten by using get_train_data() and get_test_data() methods respectively.
        self.get_train_data(train_path)
        self.X_test = self.get_test_data(test_path)
        
    def get_train_data(self, train_path):
        # Reads and prepares train data set.
        train_data = pd.read_csv(train_path)
        
        X_raw = (train_data.iloc[:, 1:].values).astype('float32')
        self.y_raw = train_data.iloc[:, 0].values.astype('int32')
        
        self.X = X_raw.reshape(X_raw.shape[0], 28, 28, 1)
        self.y = to_categorical(self.y_raw)
        
    def get_test_data(self, test_path):
        # Reads and prepares test data set.
        test_data = pd.read_csv(test_path)
        X_test = test_data.values.astype('float32')  
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        return X_test

    def upgrade(self, mult):
        # Creates a train data set via concatenating morphing initial images with slight rotation and shifting mult times.
        # Returns a tuple of a train data set and a labels vector.
        X_upg = np.array([warp(item, SimilarityTransform(translation=(np.random.randint(-1, 1), np.random.randint(-1, 1)))) 
                for item in [rotate(item, np.random.randint(-15, 15)) 
                for item in self.X]])
        y_upg = self.y
        for i in range(mult - 1):
            X_upg = np.concatenate((X_upg, 
                                    np.array([warp(item, SimilarityTransform(translation=(np.random.randint(-1, 1), np.random.randint(-1, 1)))) 
                                    for item in [rotate(item, np.random.randint(-15, 15)) 
                                    for item in self.X]])))
            y_upg = np.concatenate((y_upg, self.y))
            print('iteration {0} is sucsessfully done'.format(i))
        return X_upg, y_upg

    def fit(self, Xy, to_continue=False, load_weights=False, save_weights=False, load_path='model/weights.h5', save_path='model/weights.h5'):
        # Learns a neural net.
        # Input Xy must be a tuple of train data set and lebels vector. In this model input intends to be an upgrage() method
        # Load_weights and save_weights direct whether to load existing weights file and save resulting weights as a file respectively.
        # If load_weights and save_weights are both True, the net will be learned again with initial loaded weights.
        if load_weights:
            self.model.load_weights(load_path)
        if not load_weights or load_weights and to_continue:
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
            hist = self.model.fit(Xy[0], Xy[1], validation_split=0.1, epochs=10, callbacks=[callback])
            print(hist.history)

        if save_weights:
            self.model.save_weights(save_path)

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=-1)

    def verify(self):
        # Returns a fraction of right predictions of original training data set
        try:
            acc = self.predict(self.X)
        except Exception as e:
            print(e)
            return
        try:
            print(metrics.accuracy_score(self.y_raw, acc))
        except Exception as e:
            print(e)

    def output(self, data, file_name):
        # Creates a .csv file ready for Kaggle submission
        sample_submission = pd.read_csv("data/sample_submission.csv")
        sample_submission['Label'] = data
        sample_submission.to_csv(file_name + '.csv', index=False)