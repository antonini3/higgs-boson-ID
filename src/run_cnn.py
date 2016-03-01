
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




def run_cnn(model, learning_rate=1e-4, batch_size=32, epochs=20, dropout=1.0, print_every=50, plot=False, train_size=None):
    '''
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
        val_acc = np.mean([accuracy.eval(feed_dict={model.x: [x_i], model.y_: [y_i], model.keep_prob: 1.0}) for x_i, y_i in zip(x_val, y_val)])
        print("  Validation accuracy: %g"%val_acc)
        accuracies.append(val_acc)
        if j > 1:
            average_last_accuracy = np.mean([accuracies[j], accuracies[j-1]])
            print "  Average Validation Accuracy from epoch {0} and {1}: {2}".format(j, j-1, average_last_accuracy)
            accuracy_difference = abs(average_last_accuracy - val_acc)
            print "  Difference in accuracies: {0}".format(accuracy_difference)
            if accuracy_difference < 0.008:
                learning_rate *= 0.5
                print "  new learning rate: {0}".format(learning_rate)

    print("Final test accuracy: %g"%accuracy.eval(feed_dict={model.x: x_test, model.y_: y_test, model.keep_prob: 1.0}))

    if plot:
        our_plot(range(1, epochs+1), [accuracies])
    '''


if __name__ == '__main__':
    train_size = None
    x_train, x_test, x_val, y_train, y_test, y_val = cnn_preprocessing(train_size=train_size)
    model = LaNet()
    model.fit(x_train, y_train, x_val, y_val, learning_rate=1e-4, batch_size=32, dropout=1, decay=0.90, print_every=200, max_epochs=20)
    print "Training Accuracy: ", model.score(x_train, y_train)
    print "Validation Accuracy: ", model.score(x_val, y_val)
    print "Testing Accuracy: ", model.score(x_test, y_test)

    predictions = model.predict(x_test)

    prob_one = [pred[1] for pred in predictions]
    y_test_true = [np.argmax(y_i) for y_i in y_test]
    fpr, tpr, threshold = roc_curve(y_test_true, prob_one)
    print(fpr)
    print(tpr)
    plot_roc(y_test_true, prob_one, name='LaNet')
    plt.show()
    '''
    simple_model = SimpleModel()
    model = LaNet()
    run_cnn(model, epochs=200, learning_rate=1e-4, plot=True, print_every=200, dropout=0.5)
    '''
