
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


def classify(clf, pull=False, name=None, fine=False, train_size=None, subset_size=None):
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
	plot_roc(y_train, y_probs_train, name)

	return accuracy_test, accuracy_train

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


def preprocessing(pull=False, fine=False, train_size=None):
	higgs_data, not_higgs_data = get_data(pull, fine)

	higgs_data = zip(higgs_data, [1] * len(higgs_data))
	not_higgs_data = zip(not_higgs_data, [0] * len(not_higgs_data))

	all_data = permute_arrays(higgs_data, not_higgs_data)

	if train_size is not None and train_size < len(all_data) and train_size > 0:
		all_data = all_data[:train_size]

	x, y = zip(*all_data)
	return x, y

def opencv_preprocessing(directory):
	image_files = [directory + f for f in listdir(directory) if isfile(join(directory,f))]
	x = []
	y = []
	for image in image_files:
		img = cv2.imread(image)
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		kp, des = cv2.SURF(400).detectAndCompute(gray,None)
		if des is not None:
			x.append(des[0])
			if 'not' in image:
				y.append(0)
			else:
				y.append(1)

	return x, y

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


if __name__ == '__main__':
	# error_plotting()
	
	setup_figure()

	# # for pull, fine, name, classifier in zip([True, False, False], [False, False, True], ["Pull classifier", "Our classifier on coarse data", "Our classifier on fine data"], \
	# # 	[QuadraticDiscriminantAnalysis(), AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=50)), \
	# # 	AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=50))]):
	# # 	classify(classifier, pull=pull, name=name, fine=fine)
	# for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None]:
	# 	print "depth", depth
	#  	classify(AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=depth, n_estimators=50, max_features=100)), pull=False, name=str(depth) + " maximum depth", fine=False)
	# plot_show()

	for train_size in [1000]:
		print "train_size", train_size
	 	classify(AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=50, max_features=100)), pull=False, name=str(train_size) + " training size", fine=False, train_size=train_size)
	plot_show()


	# classifiers = [ LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), KNeighborsClassifier(), GaussianNB(), RandomForestClassifier(), 
	#     ExtraTreesClassifier(random_state=42, criterion='entropy', n_estimators=400, verbose=1), LogisticRegression(),
	#     AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=50))]
	# names = [ "LDA", "QDA", "K-Neighbors", "Gaussian Naive Bayes", "Random Forest", "Extra Trees", "Logistic Regression", "AdaBoost"]
	# classifiers = [AdaBoostClassifier(n_estimators=350, random_state=42, base_estimator=RandomForestClassifier(random_state=600, n_estimators=350))] # A = 0.800, AUC = 0.87
	# classifiers = [ExtraTreesClassifier(random_state=42, criterion='entropy', n_estimators=400, verbose=1)] 
	# classifiers = [AdaBoostRegressor(random_state=42, loss='square', n_estimators=200, base_estimator=RandomForestClassifier(random_state=600, verbose=1, n_estimators=200))] 

	# classifiers = [QuadraticDiscriminantAnalysis()]

	# names = ["Fisher Discriminant Analysis with Pull Data"]
	# setup_figure()
	# for classifier, name in zip(classifiers, names):
	# 	print name
	# 	classify(classifier, pull=True, name=name, fine=False)
	# plot_show()
	classifier = AdaBoostClassifier(n_estimators=50, base_estimator=RandomForestClassifier(max_depth=5, n_estimators=50))
	#show_features(classifier, pull=False, fine=False)
	#visualize_plot(get_parameters(classifier, pull=False, fine=False))
	component_analysis(ca_type='ica', pull=False, fine=False)
	#classify(classifier, pull=False, fine=False, name="Adaboost")
	plot_show()

	
	
	# opencv_preprocessing('../images/')


