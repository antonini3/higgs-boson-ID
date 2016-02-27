from constants import *
from util import *
from analysis import *
from classifier import *
import tensorflow as tf
import numpy as np

class CNNModel(object):

    def __init__(self):
        pass

        # Weight initialization
    def _weight_variable(self, shape):
        fan_in = shape[0] * shape[1] * shape[2] if len(shape) > 2 else shape[0]
        weight_init = 1. / np.sqrt(fan_in / 2.)
        initial = tf.truncated_normal(shape, stddev=weight_init)
        return tf.Variable(initial)

    # Bias initialization
    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Convolution
    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1., 1., 1., 1.], padding='SAME')

    # Max pool
    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class SimpleModel(CNNModel):

    def __init__(self):
        self.x = tf.placeholder("float", shape=[None, PADDED_IMAGE_DIMENSION])
        self.y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])

        # Batchsize, width/height, color channels
        x_image = tf.reshape(self.x, [-1, PADDED_NUM_PIXELS, PADDED_NUM_PIXELS, 1])

        ## 5x5 image patches, 1 color channel, 32 outputs
        W_conv1a = self._weight_variable([5, 5, 1, 32])
        b_conv1a = self._bias_variable([32])
        h_conv1a = tf.nn.relu(self._conv2d(x_image, W_conv1a) + b_conv1a)

        h_pool1 = self._max_pool_2x2(h_conv1a)

        # Second layer
        W_conv2a = self._weight_variable([5, 5, 32, 64])
        b_conv2a = self._bias_variable([64])
        h_conv2a = tf.nn.relu(self._conv2d(h_pool1, W_conv2a) + b_conv2a)


        h_pool2 = self._max_pool_2x2(h_conv2a)

        ## densely connected
        W_fc1 = self._weight_variable([8 * 8 * 64, 1024])
        b_fc1 = self._bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self._weight_variable([1024, NUM_CLASSES])
        b_fc2 = self._bias_variable([NUM_CLASSES])

        self.y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

class LaNet(CNNModel):

    def __init__(self):
        self.x = tf.placeholder("float", shape=[None, PADDED_IMAGE_DIMENSION])
        self.y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])
        self.keep_prob = tf.placeholder("float")

        # Batchsize, width/height, color channels
        x_image = tf.reshape(self.x, [-1, PADDED_NUM_PIXELS, PADDED_NUM_PIXELS, 1])

        # First Layer: (CONV3-32) x 2 - MAXPOOL
        h_conv1a = tf.nn.relu(self._conv2d(x_image, self._weight_variable([3, 3, 1, 32])) + self._bias_variable([32]))
        h_conv1b = tf.nn.relu(self._conv2d(h_conv1a, self._weight_variable([3, 3, 32, 32])) + self._bias_variable([32]))
        h_pool1 = self._max_pool_2x2(h_conv1b)

        # Second Layer: (CONV3-64) x 2 - MAXPOOL
        h_conv2a = tf.nn.relu(self._conv2d(h_pool1, self._weight_variable([3, 3, 32, 64])) + self._bias_variable([64]))
        h_conv2b = tf.nn.relu(self._conv2d(h_conv2a, self._weight_variable([3, 3, 64, 64])) + self._bias_variable([64]))
        h_pool2 = self._max_pool_2x2(h_conv2b)

        # Third Layer: (CONV3-128) x 3 - MAXPOOL
        h_conv3a = tf.nn.relu(self._conv2d(h_pool2, self._weight_variable([3, 3, 64, 128])) + self._bias_variable([128]))
        h_conv3b = tf.nn.relu(self._conv2d(h_conv3a, self._weight_variable([3, 3, 128, 128])) + self._bias_variable([128]))
        h_conv3c = tf.nn.relu(self._conv2d(h_conv3b, self._weight_variable([3, 3, 128, 128])) + self._bias_variable([128]))
        h_pool3 = self._max_pool_2x2(h_conv3c)

        # Fourth Layer: (CONV3-256) x 3 - MAXPOOL
        h_conv4a = tf.nn.relu(self._conv2d(h_pool3, self._weight_variable([3, 3, 128, 256])) + self._bias_variable([256]))
        h_conv4b = tf.nn.relu(self._conv2d(h_conv4a, self._weight_variable([3, 3, 256, 256])) + self._bias_variable([256]))
        h_conv4c = tf.nn.relu(self._conv2d(h_conv4b, self._weight_variable([3, 3, 256, 256])) + self._bias_variable([256]))
        h_pool4 = self._max_pool_2x2(h_conv4c)

        # Fully Connected: 4096 - 4096 - 1000
        h_pool4_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, self._weight_variable([2 * 2 * 256, 4096])) + self._bias_variable([4096]))
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self._weight_variable([4096, 4096])) + self._bias_variable([4096]))
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, self._weight_variable([4096, 1024])) + self._bias_variable([1024]))
        h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob)

        # Softmax
        self.y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, self._weight_variable([1024, NUM_CLASSES])) + self._bias_variable([NUM_CLASSES]))

        