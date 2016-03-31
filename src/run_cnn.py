# coding: utf-8

## Setup
from constants import *
from util import *
from analysis import *
from classifier import *
from cnn_models import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc



np.set_printoptions(threshold='nan')

def cnn_preprocessing(train_size=None):
    half_train = train_size / 2 if train_size is not None else None
    higgs_data, not_higgs_data = read_higgs_data(HIGGS_FILE_NAME, max_size=half_train, skip=6), read_higgs_data(NOT_HIGGS_FILE_NAME, max_size=half_train, skip=6)

    all_data = permute_arrays(zip(higgs_data, [1] * len(higgs_data)), zip(not_higgs_data, [0] * len(not_higgs_data)))

    x, y = zip(*all_data)

    x_total, y_total = np.asarray(x), np.asarray(y)

    # Padding
    x_temp = x_total.reshape((x_total.shape[0], NUM_PIXELS, NUM_PIXELS))
    x_temp = np.pad(x_temp, pad_width=[(0,0), (3,4), (3,4)], constant_values=0., mode='constant')
    x_total = x_temp.reshape((x_total.shape[0], 32 * 32))

    # Splitting
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=VALTEST_SET_RATIO, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=VAL_SET_RATIO, random_state=420)

    # Zero centering with training mean
    train_mean = np.mean(x_train, axis=0)
    x_train -= train_mean
    x_test -= train_mean
    x_val -= train_mean

    # Normalizing
    train_std = np.std(x_train, axis=0)
    x_train /= np.max(np.abs(x_train))
    x_test /= np.max(np.abs(x_test))
    x_val /= np.max(np.abs(x_val))

    y_train, y_test, y_val = boolean_y(y_train), boolean_y(y_test), boolean_y(y_val)

    return x_train, x_test, x_val, y_train, y_test, y_val

def get_roc_data(predictions):
    prob_one = [pred[1] for pred in predictions]
    y_test_true = [np.argmax(y_i) for y_i in y_test]
    fpr, tpr, threshold = roc_curve(y_test_true, prob_one)
    auc_ = auc(fpr, tpr)
    return fpr, tpr, auc_

if __name__ == '__main__':
    train_size = 10000
    x_train, x_test, x_val, y_train, y_test, y_val = cnn_preprocessing(train_size=train_size)
    model = LaNet()
    model.fit(x_train, y_train, x_val, y_val, learning_rate=1e-4, batch_size=32, dropout=1.0, decay=.9, print_every=200, max_epochs=30, save_file=True)
    # model = LaNet()
    model = SimpleModel()
    # print mc[0][1]
    model.fit(x_train, y_train, x_val, y_val, learning_rate=1e-4, batch_size=32, dropout=1.0, decay=.9, print_every=50, max_epochs=30)
    # mc = model.get_misclassification(x_train, y_train)
    # np.save(open('../images/misclass', 'wb'), np.asarray(mc))
    # mc = np.load(open(open('../images/misclass', 'wb')))
    # for i in range(len(mc)):
    #     print np.argmax(mc[i][1])
    #     visualize_plot(mc[i][0])

    print "Training Accuracy: ", model.score(x_train, y_train)
    print "Validation Accuracy: ", model.score(x_val, y_val)
    print "Testing Accuracy: ", model.score(x_test, y_test)
    
    '''
    ckpt = tf.train.get_checkpoint_state("../models/")
    if ckpt and ckpt.model_checkpoint_path:
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    '''
    predictions = model.predict(x_test)
    fpr, tpr, auc_ = get_roc_data(predictions)
    print(fpr)
    print(tpr)
    print(auc_)




