from sklearn.svm import SVC
from sklearn.datasets import make_classification

import os
import copy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
import numpy as np
from sklearn import cluster
from scipy.spatial import distance
import pandas as pd
from keras.utils import np_utils
import gpflow as gpf
from sklearn.metrics import f1_score


def compute_scores(flat_true, flat_pred):
    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    print("F1-BAD: ", f1_bad, "F1-OK: ", f1_good)
    print("F1-score multiplied: ", f1_bad * f1_good)
def resampleFile():
    filename = open("train.revised", "w")
    file = open("train", "r")
    for x in file:
        x = x.strip()
        filename.write(x+"\n")
        if x.endswith(",0"):
            #filename.write(x+"\n")
            filename.write(x+"\n")
    filename.close()
    file.close()
def standardize_data(X_train, X_test, X_valid):
    unique_X_train = np.unique(X_train, axis=0)
    X_mean = np.mean(unique_X_train, axis=0)
    #print(X_mean)
    X_std = np.std(unique_X_train, axis=0)+0.0000001 #a small noise
    #print(X_std)
    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid


dataset = np.loadtxt("test", delimiter=",")

x_test = dataset[:,0:144]
y_test = dataset[:,144].reshape(-1,1)
    #print(x_test[20])

dataset = np.loadtxt("dev", delimiter=",")
x_valid = dataset[:,0:144]
y_valid = dataset[:,144].reshape(-1,1)

resampleFile()

dataset = np.loadtxt("train.revised", delimiter=",")
x_train = dataset[:,0:144]
y_train = dataset[:,144].reshape(-1,1)

    
x_train_root = x_train
x_valid_root = x_valid

x_train, x_test, x_valid = standardize_data(copy.deepcopy(x_train_root), x_test, copy.deepcopy(x_valid_root))


clf = SVC()

print(y_train)
print(y_train.ravel())

clf.fit(x_train, y_train.ravel())

z = clf.predict(x_test)
print(z)

compute_scores(z, y_test.ravel())