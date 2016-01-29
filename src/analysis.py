

from constants import *
from util import *

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from os import listdir
from os.path import isfile, join

from sklearn.metrics import accuracy_score, roc_auc_score

import numpy

import itertools


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, FastICA

def get_parameters(clf, pull=False, fine=False):
	x, y = preprocessing(pull=pull, fine=fine)
	clf.fit(x, y)
	decision_function = clf.decision_function(x)
	return decision_function

def component_analysis(ca_type='pca', pull=False, fine=False):
	x, y = preprocessing(pull=pull, fine=fine)
	if ca_type == 'pca':
		ca = PCA(n_components=2)
	elif ca_type == 'ica':
		ca = FastICA(n_components=2)
	x_red= ca.fit_transform(x)
	scatter_plot(x_red, y)


def show_features(clf, pull=False, fine=False):
	x, y = preprocessing(pull=pull, fine=fine)
	clf.fit(x, y)
	important_features = clf.feature_importances_
	indices = np.argsort(important_features)[::-1][:1]
	arr = numpy.zeros(shape=(NUM_PIXELS*NUM_PIXELS,))
	for idx in indices:
		arr[idx] = 1
	#convert_matrix_to_jpg(arr, 'ten_imp_features.jpg')
	visualize_plot(arr)

	# matrix_indices = [array_index_to_matrix_index(array_index) for array_index in indices]
	# mat = numpy.zeros(shape=(NUM_PIXELS,NUM_PIXELS))
	# for matrix_idx in matrix_indices:
	# 	mat[matrix_idx[0]:matrix_idx[1]] = 1
	# print mat

def plot_best_features(clf, x, y):
	clf.fit(x, y)
	important_features = clf.feature_importances_
	indices = np.argsort(important_features)[::-1]
	all_accuracy = []
	for i in xrange(len(indices)):
		x_subset = []
		for x_i in x:
			x_i_subset = []
			for j, idx in enumerate(indices):
				x_i_subset.append(x_i[idx])
				if j==i:
					break
			x_subset.append(x_i_subset)
		x_train, x_test, y_train, y_test = train_test_split(x_subset, y, test_size=TEST_SET_RATIO, random_state=42)
		clf.fit(x_train, y_train)
		y_hat = clf.predict(x_test)
		accuracy_test = accuracy_score(y_test, y_hat)
		all_accuracy.append(accuracy_test)
		if i == 40:
			break
	plot_accuracy(all_accuracy)


def error_plotting():
	all_size = [30, 50, 100, 500, 1000, 1500, 2000, 3000, 5000, 10000]
	# all_size = [50,100]
	train_size = [elem * (1 - TEST_SET_RATIO) for elem in all_size]
	errors = [classify(AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=50, max_features=100)), pull=False, fine=False, train_size=size) for size in all_size]

	# classifier = AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=50))
	# subset_features = 40
	# errors = [classify(classifier, pull=False, name="{0} features and {1} training set size".format(subset_features, size),fine=False, train_size=size, subset_size=subset_features) for size in all_size]

	test_error, train_error = zip(*errors)
	plot_error(train_size, test_error, train_error)

def plot_trees(clf, pull=False, fine=False):
	x, y = preprocessing(pull=pull, fine=fine)
	clf.fit(x, y)

	def rules(clf, features, labels, node_index=0):
	    """Structure of rules in a fit decision tree classifier

	    Parameters
	    ----------
	    clf : DecisionTreeClassifier
	        A tree that has already been fit.

	    features, labels : lists of str
	        The names of the features and labels, respectively.

	    from http://planspace.org/20151129-see_sklearn_trees_with_d3/
	    """
	    node = {}
	    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
	        count_labels = zip(clf.tree_.value[node_index, 0], labels)
	        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
	                                  for count, label in count_labels))
	    else:
	        feature = features[clf.tree_.feature[node_index]]
	        threshold = clf.tree_.threshold[node_index]
	        node['name'] = '{} > {}'.format(feature, threshold)
	        left_index = clf.tree_.children_left[node_index]
	        right_index = clf.tree_.children_right[node_index]
	        node['children'] = [rules(clf, features, labels, right_index),
	                            rules(clf, features, labels, left_index)]
	   	return node
	print NUM_PIXELS*NUM_PIXELS
	print []
	clf_rules = rules(clf, ["Pixel ({0}, {1})".format(str(array_index_to_matrix_index(i))) for i in range(NUM_PIXELS*NUM_PIXELS)], ["Higgs", "Non-Higgs"])
	print clf_rules
