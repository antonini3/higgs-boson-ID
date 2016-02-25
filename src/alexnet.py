
# coding: utf-8

# In[1]:

# AlexNet implementation example using TensorFlow library.
# This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
# AlexNet Paper (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

# Author: Aymeric Damien
# Project: https://github.com/aymericdamien/TensorFlow-Examples/


# In[43]:

from constants import *
from util import *
from analysis import *
from classifier import *
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import logging, datetime
# logging.basicConfig(filename='/farmshare/user_data/antonaf/alexnet ' + str(datetime.datetime.now()) + '.log',level=logging.DEBUG)


# In[4]:

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[6]:

x_total, y_total = preprocessing(pull=False, fine=False)
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=TEST_SET_RATIO, random_state=42)

max_x = max(x_train.max(), x_train.max())
x_train *= 1.0 / max_x
x_test *= 1.0 / max_x


y_train = boolean_y(y_train)
y_test = boolean_y(y_test)


# In[7]:

# Parameters
learning_rate = 0.001
training_iters = 300000
batch_size = 64
display_step = 100


# In[28]:

# Network Parameters
n_pixels = NUM_PIXELS
n_input = IMG_DIMENSION # MNIST data input (img shape: 28*28)
n_classes = NUM_CLASSES # MNIST total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units


# In[10]:

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder("float") # dropout (keep probability)


# In[29]:

# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], 
                                                  padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], 
                          padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 25, 25, 1])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    # Reshape conv3 output to fit dense layer input
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    # Relu activation
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    
    # Relu activation
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') 

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out


# In[30]:

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 512])),
    'out': tf.Variable(tf.random_normal([512, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[31]:

# Construct model
pred = alex_net(x, weights, biases, keep_prob)


# In[32]:

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[33]:

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[39]:

# Initializing the variables
init = tf.initialize_all_variables()


# In[44]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    n_epochs = 10
    # Keep training until reach max iterations

    for j in xrange(n_epochs):
        print "Epoch number:", j + 1
        logging.info("Epoch number:" + str(j + 1))
        num_batches = max(x_train.shape[0] // batch_size, 1)
        x_epoch, y_epoch = permute(x_train, y_train)
        logging.info("  Number of batches:", num_batches)
        print "  Number of batches:", num_batches
        for i in xrange(num_batches):
            x_batch, y_batch = x_epoch[i * batch_size : (i + 1) * batch_size], y_epoch[i * batch_size : (i + 1) * batch_size]
            assert(len(x_batch) != 0)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, keep_prob: dropout})
            if i % 10 == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.})
                out_str = "    Iter " + str(i) + ", Minibatch Loss= "                       + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
                print out_str
                logging.info(out_str)
        test_acc = "  Test accuracy %g"%accuracy.eval(feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
        logging.info(test_acc)
        print test_acc
    logging.info("Optimization Finished!")
    print "Optimization Finished!"
    
    # Calculate accuracy for 256 mnist test images
    test_acc = sess.run(accuracy, feed_dict={x: x_test, 
                                                             y: y_test, 
                                                             keep_prob: 1.})
    logging.info(test_acc)
    print test_acc


# In[ ]:



