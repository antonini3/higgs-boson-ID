
# coding: utf-8

## Setup
from constants import *
from util import *
from analysis import *
from classifier import *
import tensorflow as tf
import numpy as np


# Multilayer convolutional neural network

## weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


## convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1., 1., 1., 1.], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## build the computation graph

x = tf.placeholder("float", shape=[None, IMG_DIMENSION])
y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])

# Batchsize, width/height, color channels
x_image = tf.reshape(x, [-1, NUM_PIXELS, NUM_PIXELS, 1])

## 5x5 image patches, 1 color channel, 32 outputs
W_conv1a = weight_variable([5, 5, 1, 32])
b_conv1a = bias_variable([32])
h_conv1a = tf.nn.relu(conv2d(x_image, W_conv1a) + b_conv1a)

h_pool1 = max_pool_2x2(h_conv1a)

# Second layer
W_conv2a = weight_variable([5, 5, 32, 64])
b_conv2a = bias_variable([64])
h_conv2a = tf.nn.relu(conv2d(h_pool1, W_conv2a) + b_conv2a)


h_pool2 = max_pool_2x2(h_conv2a)

## densely connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, NUM_CLASSES])
b_fc2 = bias_variable([NUM_CLASSES])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

NUM_EPOCHS = 50
batch_size = 30

## train and evaluate the model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-7))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
training_sizes = [7500, 10000]
all_test_accuracies = []
fig = plt.figure()

for train_size in training_sizes:
    ## Start interactiveSession
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    test_accuracies = []
    x_total, y_total = preprocessing(pull=False, fine=False, train_size = train_size)
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=TEST_SET_RATIO, random_state=42)
    max_x = max(x_train.max(), x_train.max())
    x_train *= 1.0 / max_x
    x_test *= 1.0 / max_x
    y_train = boolean_y(y_train)
    y_test = boolean_y(y_test)

    for j in xrange(NUM_EPOCHS):
        print "Epoch number:", j + 1
        num_batches = max(x_train.shape[0] // batch_size, 1)
        x_epoch, y_epoch = permute(x_train, y_train)
        print "  Number of batches:", num_batches
        for i in xrange(num_batches):
            x_batch, y_batch = x_epoch[i * batch_size : (i + 1) * batch_size], y_epoch[i * batch_size : (i + 1) * batch_size]
            assert(len(x_batch) != 0)
            if i%10 == 0:
                train_accuracy = accuracy.eval(feed_dict={ x:x_batch, y_: y_batch, keep_prob: 1.0})
                print("     Step: %d, Training accuracy: %f"%(i, train_accuracy))
            train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
        test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
        print("Test accuracy %g" % test_accuracy)
        test_accuracies.append(test_accuracy)

    plt.plot(range(1, NUM_EPOCHS +1), test_accuracies, label=str(train_size * (1-TEST_SET_RATIO)))
    print("FINAL TEST ACCURACY: %g"%accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
    sess.close()

plt.legend(loc='lower right')
plt.show()

