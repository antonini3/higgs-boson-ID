
# coding: utf-8

## Setup
from constants import *
from util import *
from analysis import *
from classifier import *
from cnn_models import *
import tensorflow as tf
import numpy as np

np.set_printoptions(threshold='nan')

def cnn_preprocessing(train_size=None):
    x_total, y_total = preprocessing(pull=False, fine=False)

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




def run_cnn(model, learning_rate=1e-4, batch_size=32, epochs=20, dropout=1.0, print_every=50, plot=False, train_size=None):
    sess = tf.InteractiveSession()
    x_train, x_test, x_val, y_train, y_test, y_val = cnn_preprocessing(train_size=train_size)

    cross_entropy = -tf.reduce_sum(model.y_*tf.log(model.y_conv + 1e-7))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(model.y_conv,1), tf.argmax(model.y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    accuracies = []

    for j in xrange(epochs):
        print "Epoch number:", j + 1
        num_batches = max(x_train.shape[0] // batch_size, 1)
        x_epoch, y_epoch = permute(x_train, y_train)
        print "  Number of batches:", num_batches
        for i in xrange(num_batches):
            x_batch, y_batch = x_epoch[i * batch_size : (i + 1) * batch_size], y_epoch[i * batch_size : (i + 1) * batch_size]
            assert(len(x_batch) != 0)
            if i > 0 and i % print_every == 0:
                train_accuracy = accuracy.eval(feed_dict={ model.x:x_batch, model.y_: y_batch, model.keep_prob: 1.0})
                print("     Step: %d, Training accuracy: %f"%(i, train_accuracy))
            train_step.run(feed_dict={model.x: x_batch, model.y_: y_batch, model.keep_prob: dropout})
        val_acc = accuracy.eval(feed_dict={model.x: x_val, model.y_: y_val, model.keep_prob: 1.0})
        accuracies.append(val_acc)
        print("  Validation accuracy: %g"%val_acc)

    print("Final test accuracy: %g"%accuracy.eval(feed_dict={model.x: x_test, model.y_: y_test, model.keep_prob: 1.0}))

    if plot:
        our_plot(range(1, epochs+1), [accuracies])



cnn_preprocessing()

if __name__ == '__main__':
    simple_model = SimpleModel()
    model = LaNet()
    run_cnn(model, epochs=2, plot=True, print_every=20, dropout=0.8)

