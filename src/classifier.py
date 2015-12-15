
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


def classify(clf, pull=False, name=None, fine=False, train_size=None):
	x, y = preprocessing(pull=pull, fine=fine)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SET_RATIO, random_state=42)
	clf.fit(x_train, y_train)
	y_hat = clf.predict(x_test)
	accuracy_test = accuracy_score(y_test, y_hat)

	y_hat_train = clf.predict(x_train)
	accuracy_train = accuracy_score(y_train, y_hat_train)
	print "Accuracy is: {0}".format(accuracy_test)
	y_probs_test = clf.predict_proba(x_test)

	# y_probs_train = clf.predict_proba(x_train)
	plot_roc(y_test, y_probs_test, name)

	return accuracy_test, accuracy_train






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
	test_error, train_error = zip(*errors)
	plot_error(train_size, test_error, train_error)


if __name__ == '__main__':
	# error_plotting()
	# classify(svm.SVC(verbose=1, kernel='poly', max_iter=100000000))
	
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
	
	
	
	# opencv_preprocessing('../images/')


