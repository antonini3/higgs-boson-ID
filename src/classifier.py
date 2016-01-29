

from constants import *
from util import *
from analysis import *

from sklearn.cross_validation import train_test_split, KFold
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

''' Our final 229 classifier: AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=11, n_estimators=50, max_features=100)) '''

def simple_classify(clf, pull=False, name=None, fine=False, train_size=None, subset_size=None):
	x, y = preprocessing(pull=pull, fine=fine)

	if subset_size != None:
		clf.fit(x, y)
		important_features = clf.feature_importances_
		indices = np.argsort(important_features)[::-1]
		x_subset = []
		for x_i in x:
			x_i_subset = []
			for j, idx in enumerate(indices):
				x_i_subset.append(x_i[idx])
				if j == subset_size:
					break
			x_subset.append(x_i_subset)
		x = x_subset

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SET_RATIO, random_state=42)
	clf.fit(x_train, y_train)

	y_hat = clf.predict(x_test)
	accuracy_test = accuracy_score(y_test, y_hat)

	y_hat_train = clf.predict(x_train)
	accuracy_train = accuracy_score(y_train, y_hat_train)
	print "Accuracy is: {0}".format(accuracy_test)
	y_probs_test = clf.predict_proba(x_test)

	y_probs_train = clf.predict_proba(x_train)
	plot_roc(y_test, y_probs_test, name)

	return accuracy_test, accuracy_train

def cross_validate(clf, pull=False, name=None, fine=False, k=2):
	x, y = preprocessing(pull=pull, fine=fine)
	kf = KFold(len(x), n_folds=k)
	auc_sum = 0.0
	for i, (train_indices, test_indices) in enumerate(kf):
		x_train, x_test = x[train_indices], x[test_indices]
		y_train, y_test = y[train_indices], y[test_indices]
		clf.fit(x_train, y_train)
		y_probs = clf.predict_proba(x_test)
		auc = get_auc(y_test, y_probs)
		auc_sum += auc
		print "Iteration number {0} has auc of {1}".format(i, auc)

	print "Average AUC is: {0}".format(auc_sum/k)

def correlation_image(clf, pull=False, name=None, fine=False):
	x, y = preprocessing(pull=pull, fine=fine)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SET_RATIO, random_state=42)
	clf.fit(x, y)
	decision_function = clf.decision_function(x)
	correlations = []
	for i, x_i in enumerate(x):
		corr_i = scipy.stats.pearsonr(x_i, decision_function)
		correlations.append(corr_i)
	visualize_plot(correlations)
		
def preprocessing(pull=False, fine=False, train_size=None):
	higgs_data, not_higgs_data = get_data(pull, fine)

	higgs_data = zip(higgs_data, [1] * len(higgs_data))
	not_higgs_data = zip(not_higgs_data, [0] * len(not_higgs_data))

	all_data = permute_arrays(higgs_data, not_higgs_data)

	if train_size is not None and train_size < len(all_data) and train_size > 0:
		all_data = all_data[:train_size]

	x, y = zip(*all_data)
	return np.asarray(x), np.asarray(y)


if __name__ == '__main__':

	# setup_figure()

	classifier = AdaBoostClassifier(n_estimators=100, base_estimator=RandomForestClassifier(max_depth=11, n_estimators=50, max_features=100))
	crappy_classifier = AdaBoostClassifier(n_estimators=1, base_estimator=RandomForestClassifier(max_depth=1, n_estimators=1))
	# simple_classify(classifier, pull=False, fine=False, name="Adaboost")
	cross_validate(classifier, k=5)
	# correlation_image(classifier)


	# plot_show()


