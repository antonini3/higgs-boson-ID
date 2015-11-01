
from constants import *

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import scipy




def visualize_plot(arr):
	matrix = array_to_numpy_matrix(arr)
	fig = plt.figure()
	plt.imshow(matrix)
	plt.show()


def array_to_matrix(arr):
	return [[arr[i * NUM_PIXELS + j] for j in range(NUM_PIXELS)] for i in range(NUM_PIXELS)]

def array_to_numpy_matrix(arr):
	return np.matrix(array_to_matrix(arr)).astype(float)

def read_data(filename):
	return np.asarray([np.asarray(x.split()).astype(float) for x in open(filename)])

def permute_arrays(arr1, arr2):
	return np.random.permutation(np.concatenate((arr1, arr2)))

def plot_roc(y_test, y_probs):
	print y_probs
	fpr, tpr, threshold = roc_curve(y_test, y_probs[:,1])
	roc_auc = auc(fpr, tpr)
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

def plot_error(x, first_error, second_error=None):
	plt.figure()
	plt.plot(x, first_error)
	if second_error is not None:
		plt.plot(x, second_error)
	plt.show()

def convert_matrix_to_jpg(array, jpg_file_name):
	scipy.misc.toimage(array_to_numpy_matrix(array), cmin=0.0).save('../images/' + jpg_file_name)

def convert_all_to_jpg():
	higgs_data = read_data(HIGGS_FILE_NAME)
	for i, higgs_matrix in enumerate(higgs_data):
		convert_matrix_to_jpg(higgs_matrix, 'higgs_{0}.jpg'.format(i))
	not_higgs_data = read_data(NOT_HIGGS_FILE_NAME)
	for i, not_higgs_matrix in enumerate(not_higgs_data):
		convert_matrix_to_jpg(higgs_matrix, 'not_higgs_{0}.jpg'.format(i))

if __name__ == '__main__':
	# higgs_file_data = read_data(NOT_HIGGS_FILE_NAME)
	# visualize_plot(higgs_file_data[0])
	convert_all_to_jpg()