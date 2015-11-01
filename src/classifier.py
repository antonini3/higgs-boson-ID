
from constants import *
from util import *

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

def classify(clf):
	x, y = preprocessing()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SET_RATIO)
	clf.fit(x_train, y_train)
	y_hat = clf.predict(x_test)

	print "Accuracy is: {0}".format(accuracy_score(y_test, y_hat))
	plot_roc(y_test, y_hat)



def preprocessing():
	higgs_data = read_data(HIGGS_FILE_NAME)
	not_higgs_data = read_data(NOT_HIGGS_FILE_NAME)

	higgs_data = zip(higgs_data, [1] * len(higgs_data))
	not_higgs_data = zip(not_higgs_data, [0] * len(not_higgs_data))

	all_data = permute_arrays(higgs_data, not_higgs_data)

	x, y = zip(*all_data)
	return x, y

if __name__ == '__main__':
	#classify(LogisticRegression(verbose=1, max_iter=1000))
	# classify(svm.SVC(verbose=1, kernel='poly', max_iter=100000000))
	classify(ExtraTreesClassifier(n_estimators=1500, verbose=1))

