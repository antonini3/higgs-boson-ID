from dataset import Dataset

from time import time
from scipy.stats import randint as sp_randint

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from os.path import dirname, join, realpath
import logging
import json, abc


logging.basicConfig(format="[%(asctime)s]\t%(msg)s", level=logging.INFO)

class Classifier(object):

    def __init__(self, dataset, verbose=False, cnn_preprocessing=False):
        self.rootpath = dirname(dirname(realpath(__file__)))
        self.modelpath = join(self.rootpath, "models")
        self.verbose = verbose
        self.dataset = dataset
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = dataset.get_data(cnn_preprocessing=cnn_preprocessing)


    def log(self, *args, **kwargs):
        if self.verbose: 
            logging.info(*args, **kwargs)


    @abc.abstractmethod
    def fit(self, **kwargs):
        """
        Fits the classifier to training data
        """
        pass


    @abc.abstractmethod
    def predict(self, data_type='val'):
        """
        Predicts ([1, 0]) for a given data type.
        Args:
            data_type:      'train', 'test' or 'val'
        """
        pass


    @abc.abstractmethod
    def probs(self, data_type='val'):
        """
        Produces probabilities for a given data type.
        Args:
            data_type:      'train', 'test' or 'val'
        """
        pass


    def score(self, score_type='accuracy', data_type='val'):
        """
        Produces a score for a given data type.
        Args:
            data_type:      'train', 'test' or 'val'
            score_type:     'accuracy', 'f1', or 'auc'
        """
        x, y = self._get_data_type(data_type)
        return self._get_score(x, y, score_type, data_type)


    def _get_data_type(self, data_type):
        if data_type == 'val':
            return self.x_val, self.y_val
        elif data_type == 'test':
            return self.x_test, self.y_test
        elif data_type == 'train':
            return self.x_train, self.y_train
        else:
            raise Exception('Inexistent data type: %s' % data_type)

    def _get_score(self, x, y, score_type, data_type):
        if score_type == 'accuracy':
            return accuracy_score(y, self.predict(data_type))
        elif score_type == 'f1':
            return f1_score(y, self.predict(data_type))
        elif score_type == 'auc':
            return roc_auc_score(y, self.probs(data_type))
        else:
            raise Exception('Inexistent score type: %s' % score_type)


class SKLClassifier(Classifier):

    def __init__(self, clf, dataset, verbose=False, cnn_preprocessing=False, **kwargs):
        super(SKLClassifier, self).__init__(dataset, verbose=verbose, cnn_preprocessing=cnn_preprocessing)
        self.log('-- SKLCLASSIFIER --')
        self.clf = clf(**kwargs)


    def fit(self, **kwargs):
        """
        Fits the classifier to training data
        """
        self.log('  Starting to learn...')
        self.clf.fit(self.x_train, self.y_train, **kwargs)


    def predict(self, data_type):
        """
        Predicts ([1, 0]) for a given data type.
        Args:
            data_type:      'train', 'test' or 'val'
        """
        self.log('  Creating predictions for %s...' % (data_type))
        x, _ = self._get_data_type(data_type)
        return self.clf.predict(x)


    def probs(self, data_type):
        """
        Produces probabilities for a given data type.
        Args:
            data_type:      'train', 'test' or 'val'
        """
        self.log('  Calculating probs for %s...' % (data_type))
        x, _ = self._get_data_type(data_type)
        return self.clf.predict_proba(x)


    def score(self, score_type = 'accuracy', data_type = 'val'):
        """
        Produces a score for a given data type.
        Args:
            data_type:      'train', 'test' or 'val'
            score_type:     'accuracy', 'f1', or 'auc'
        """
        x, y = self._get_data_type(data_type)
        return self._get_score(x, y, score_type, data_type)


    def random_search(self, param_dist, num_iters=20):
        
        random_search = RandomizedSearchCV(self.clf, param_distributions=param_dist, n_iter=num_iters, verbose=self.verbose, scoring='roc_auc')

        start = time()
        random_search.fit(self.x_train, self.y_train)
        self.log("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), num_iters))
        self._report(random_search.cv_results_)

    # Utility function to report best scores
    def _report(self, results, num_report=3):
        for i in range(1, num_report + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                self.log("Model with rank: {0}".format(i))
                self.log("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                self.log("Parameters: {0}".format(results['params'][candidate]))
                self.log("")


# class EnsembleClassifier(Classifier): 


class TensorflowUtils(object):

    def __init__(self):
        pass

    # Weight initialization
    def _weight_variable(self, shape, name=None):
        fan_in = shape[0] * shape[1] * shape[2] if len(shape) > 2 else shape[0]
        weight_init = 1. / np.sqrt(fan_in / 2.)
        initial = tf.truncated_normal(shape, stddev=weight_init)
        return tf.Variable(initial) if name is None else tf.Variable(initial, name=name)

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

    def _get_loss_function(self, loss='cross_entropy'):
        if loss == 'cross_entropy':
            self.loss_function = -tf.reduce_sum(self.y_ * tf.log(self.y + 1e-7))
            return self.loss_function

    def _get_optimizer(self, optimizer='Adam'):
        if optimizer == 'Adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = self.opt.minimize(self.loss_function)
            return self.optimizer

    def _get_minimizer(self, minimizer='accuracy'):
        if minimizer == 'accuracy':
            correct_prediction = tf.equal(tf.argmax(self.y_,1), tf.argmax(self.y,1))
            self.minimizer = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            return self.minimizer

    def _get_batches(self, x, y, batch_size):
        perm = range(len(x))
        np.random.shuffle(perm)
        x_, y_ = x[perm], y[perm]
        batches = []
        num_batches = max((x.shape[0] // batch_size) + 1, 1)
        left = 0
        right = min(batch_size, len(x))
        for i in xrange(num_batches):
            x_batch, y_batch = x_[left:right], y_[left:right]
            batches.append((x_batch, y_batch))
            left = right
            right = min((batch_size * (i + 2)), len(x))
        return batches

        
class NeuralClassifier(Classifier, TensorflowUtils):

    def __init__(self, dataset, cnn_preprocessing=False, verbose=False, learning_rate=1e-4, decay=1.0,
        batch_size=32, hidden_layers=(100,),input_shape=25*25, n_classes=2, dropout=1.0, epochs=50):
        super(NeuralClassifier, self).__init__(dataset, verbose=verbose, cnn_preprocessing=cnn_preprocessing)

        self.x = tf.placeholder('float', [None, input_shape])
        self.y = tf.placeholder('float', [None, n_classes])
        self.keep_prob = tf.placeholder('float')

        self.num_hidden = len(hidden_layers) if hidden_layers is not None else 0
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.decay = decay
        self.dropout = dropout
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs = epochs

        self._init_weights()
        self._init_network()

    def fit(self):
        cross_entropy = self._get_loss_function()
        optimizer = self._get_optimizer()
        cost = self._get_minimizer()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.log('  Starting to learn in %d epochs...' % self.epochs)
        for epoch in xrange(self.epochs):
            avg_cost = 0.
            batches = self._get_batches(self.x_train, self.y_train, self.batch_size)
            num_batches = float(len(batches))
            for x_batch, y_batch in batches:
                _, c = self.sess.run([optimizer, cost], feed_dict={self.x:x_batch, self.y:y_batch, self.keep_prob:self.dropout})
                avg_cost += c / num_batches
            self.log('  Epoch: %d, cost: %.9f' % (epoch + 1, avg_cost))
            self.learning_rate *= self.decay
        self.log('  Learning complete!')


    def predict(self, data_type):
        x, _ = self._get_data_type(data_type)
        preds = tf.argmax(self.y_, 1).eval(feed_dict={self.x: x, self.keep_prob:1.0}, session=self.sess)
        return np.asarray([[0., 1.] if p == 1 else [1., 0.] for p in preds])


    def probs(self, data_type):
        x, _ = self._get_data_type(data_type)
        return self.y_.eval(feed_dict={self.x: x, self.keep_prob:1.0}, session=self.sess)


    def _init_weights(self):
        # Initialize weights
        self.weights = {}
        in_size = self.input_shape
        if self.hidden_layers is not None:
            for i, layer in enumerate(self.hidden_layers):
                self.weights['h%d' % (i+1)] = self._weight_variable([in_size, layer]) 
                in_size = layer
        self.weights['out'] = self._weight_variable([in_size, self.n_classes])

        # Initialize biases
        self.biases = {}
        if self.hidden_layers is not None:
            self.biases = {'b%d' % (i+1) : self._bias_variable([layer]) for i, layer in enumerate(self.hidden_layers)}
        self.biases['out'] = self._bias_variable([self.n_classes])


    def _init_network(self):
        self.layers = []
        in_ = self.x
        for i in xrange(self.num_hidden):
            layer = tf.nn.relu(tf.add(tf.matmul(in_, self.weights['h%d' % (i+1)]), self.biases['b%d' % (i+1)]))
            layer_dp = tf.nn.dropout(layer, self.keep_prob)
            in_ = layer_dp
            self.layers.append(layer)
            self.layers.append(layer_dp)
        out_layer = tf.add(tf.matmul(in_, self.weights['out']), self.biases['out'])
        self.layers.append(out_layer)
        self.y_ = tf.nn.softmax(out_layer)


def test_skl():
    dataset = Dataset('rotated_sample', verbose=True)
    classifier = SKLClassifier(MLPClassifier, dataset, verbose=True, hidden_layer_sizes=(12,), max_iter=10000)
    classifier.fit()
    print 'Train auc is:\t%.3f' % classifier.score(data_type='train', score_type='auc')
    print 'Validation auc is:\t%.3f' % classifier.score(data_type='val', score_type='auc')



def test_nn():
    dataset = Dataset('rotated_sample', verbose=True)
    classifier = NeuralClassifier(dataset, verbose=True, learning_rate=5e-3, decay=0.99, hidden_layers=(12,), epochs=500)
    classifier.fit()
    print 'Train auc is:\t%.3f' % classifier.score(data_type='train', score_type='auc')
    print 'Validation auc is:\t%.3f' % classifier.score(data_type='val', score_type='auc')

if __name__ == '__main__':
    test_skl()
    test_nn()
