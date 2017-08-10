from dataset import Dataset
from classifier import *

from scipy.stats import randint as sp_randint
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from os.path import dirname, join, realpath
import logging
import json, abc


logging.basicConfig(format="[%(asctime)s]\t%(msg)s", level=logging.INFO)

class HiggsPredictor(object):

    def __init__(self, classifier):
        self.rootpath = dirname(dirname(realpath(__file__)))
        self.log('-- HIGGS PREDICTOR --')

        self.classifier = classifier

    def log(self, *args, **kwargs):
        logging.info(*args, **kwargs)

    def score(self, score_type='auc'):
        tr = self.classifier.score(data_type='train', score_type=score_type)
        va = self.classifier.score(data_type='val', score_type=score_type)
        te = self.classifier.score(data_type='test', score_type=score_type)
        self.log('Train %s is:\t%.3f' % (score_type, tr))
        self.log('Val %s is:\t%.3f' % (score_type, va))
        self.log('Test %s is:\t%.3f' % (score_type, te))
        return tr, va, te


    def classify(self):
        self.classifier.fit()
        self.score('auc')
        self.score('accuracy')


    # Source: http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
    def random_search(self, param_dist, num_iters=50):
        try:
            self.classifier.random_search(param_dist, num_iters=num_iters)
        except:
            raise Exception('Classifier does not have random search functionality.')

if __name__ == '__main__':
    dataset = Dataset('rotated_sample', verbose=True, r_restriction=None)
    classifier = SKLClassifier(MLPClassifier, dataset)
    hp = HiggsPredictor(classifier)
    hp.classify()

    # classifier = NeuralClassifier(dataset, verbose=True, learning_rate=1e-2, decay=0.99, hidden_layers=(64,), epochs=100, dropout=1.0)
    # param_dist = {"max_depth": sp_randint(12, 25),
    #           "max_features": sp_randint(6, 20),
    #           "min_samples_split": sp_randint(2, 11),
    #           "min_samples_leaf": sp_randint(1, 11),
    #           "bootstrap": [True, False],
    #           "criterion": ["gini", "entropy"]}