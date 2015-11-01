
from constants import *
from util import *

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from os import listdir
from os.path import isfile, join

from sklearn.metrics import accuracy_score, roc_auc_score

import cv2

def classify(clf, train_size=None):
	x, y = opencv_preprocessing('../images/')
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SET_RATIO, random_state=42)
	print len(x_train)
	clf.fit(x_train, y_train)
	y_hat = clf.predict(x_test)
	accuracy_test = accuracy_score(y_test, y_hat)

	y_hat_train = clf.predict(x_train)
	accuracy_train = accuracy_score(y_train, y_hat_train)
	print "Accuracy is: {0}".format(accuracy_test)
	y_probs = clf.predict_proba(x_test)

	y_probs_train = clf.predict_proba(x_train)
	plot_roc(y_train, y_probs_train)
	return accuracy_test, accuracy_train



def preprocessing(train_size=None):
	higgs_data = read_data(HIGGS_FILE_NAME)
	not_higgs_data = read_data(NOT_HIGGS_FILE_NAME)

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
	all_size = [100, 500, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
	train_size = [elem * (1 - TEST_SET_RATIO) for elem in all_size]
	errors = [classify(ExtraTreesClassifier(n_estimators=500, verbose=1), size) for size in all_size]
	test_error, train_error = zip(*errors)
	plot_error(train_size, test_error, train_error)






if __name__ == '__main__':
	# error_plotting()
	# classify(LogisticRegression(verbose=1, max_iter=300))
	# classify(svm.SVC(verbose=1, kernel='poly', max_iter=100000000))
	classify(ExtraTreesClassifier(n_estimators=300, verbose=1))
	# opencv_preprocessing('../images/')

