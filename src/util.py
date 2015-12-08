
from constants import *

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import scipy

import sys

def plot_accuracy(accuracies):
	fig = plt.figure()
	plt.plot([1 + i for i in range(len(accuracies))], accuracies)
	plt.xlabel('Number of Features')
	plt.ylabel('Accuracy')
	plt.title('Features by importance and accuracy they add')
	plt.show()

def visualize_plot(arr, title=None, x_axis=None, y_axis=None):
	matrix = array_to_numpy_matrix(arr)
	fig = plt.figure()
	if title is not None:
		plt.title(title)
	if x_axis is not None:
		plt.xlabel(x_axis)
	if y_axis is not None:
		plt.ylabel(y_axis)
	plt.imshow(matrix, cmap=plt.get_cmap('Paired'))
	plt.show()

def global_maximas(arr, THRESHOLD=0.0, NUM_AROUND=1):
	matrix = array_to_matrix(arr)

	total = 0
	for row in range(NUM_AROUND, NUM_PIXELS - NUM_AROUND):
		for col in range(NUM_AROUND, NUM_PIXELS - NUM_AROUND):
			elem = matrix[row][col]
			is_max = True
			for row_n in range(row - NUM_AROUND, row + NUM_AROUND + 1):
				for col_n in range(row - NUM_AROUND, row + NUM_AROUND + 1):
					if matrix[row_n][col_n] > elem - THRESHOLD:
						is_max = False
						break
				if not is_max:
					break
			total += int(is_max)

	return total

def array_to_matrix(arr):
	return [[arr[i * NUM_PIXELS + j] for j in range(NUM_PIXELS)] for i in range(NUM_PIXELS)]

def array_to_numpy_matrix(arr):
	return np.matrix(array_to_matrix(arr)).astype(float)

def get_data(pull=False, fine=False):
	if pull is False:
		if fine is True:
			higgs_data = read_data_without_pull(FINE_HIGGS_FILE_NAME_W_PULL)
			not_higgs_data = read_data_without_pull(FINE_NOT_HIGGS_FILE_NAME_W_PULL)
		else:
			higgs_data = read_data_without_pull(HIGGS_FILE_NAME_W_PULL)
			not_higgs_data = read_data_without_pull(NOT_HIGGS_FILE_NAME_W_PULL)
	else:
		if fine is True:
			higgs_data = read_pull(FINE_HIGGS_FILE_NAME_W_PULL)
			not_higgs_data = read_pull(FINE_NOT_HIGGS_FILE_NAME_W_PULL)
		else:
			higgs_data = read_pull(HIGGS_FILE_NAME_W_PULL)
			not_higgs_data = read_pull(NOT_HIGGS_FILE_NAME_W_PULL)
	return higgs_data, not_higgs_data

def read_data(filename):
	return np.asarray([np.asarray(x.split()).astype(float) for x in open(filename)])

def scatter_plot(x, y, dec_boundry=None):
	higgs = [x[i] for i in xrange(len(x)) if y[i] == 1]
	non_higgs = [x[i] for i in xrange(len(x)) if y[i] == 0]
	x_1_higgs, x_2_higgs = zip(*higgs)
	x_1_non_higgs, x_2_non_higgs = zip(*non_higgs)
	plt.scatter(x_1_higgs, x_2_higgs, color='green')
	plt.scatter(x_1_non_higgs, x_2_non_higgs, color='blue')

def read_data_without_pull(filename):
	if 'fine' in filename:
		l = []
		i = 0
		with open(filename) as f:
			for x in f:
				i+= 1
				a = x.split()
				l.append(np.asarray(a).astype(float)[5:])
				sys.stdout.write("Progress from {0}: {1} \r".format(filename, i))
				sys.stdout.flush()
				if i > M:
					print i
					break
		return np.asarray(l)

	elif 'withpull' in filename:
		num_skipped = 2
 		return np.asarray([np.asarray(x.split()).astype(float)[num_skipped:] for x in open(filename)])
 	else:
 		print "No pull in this file"
	
def read_pull(filename):
	if 'fine' in filename:
		l = []
		i = 0
		with open(filename) as f:
			for x in f:
				i+= 1
				l.append(np.asarray(x.split()).astype(float)[:5])
				sys.stdout.write("Progress from {0}: {1} \r".format(filename, i))
				sys.stdout.flush()
				if i > M:
					break
		return np.asarray(l)

	if 'withpull' in filename:
		return np.asarray([np.asarray(x.split()).astype(float)[:2] for x in open(filename)])
	else:
		print "No pull in this file"

def permute_arrays(arr1, arr2):
	return np.random.permutation(np.concatenate((arr1, arr2)))

def setup_figure():
	plt.figure()

def plot_roc(y_test, y_probs, name):
	fpr, tpr, threshold = roc_curve(y_test, y_probs[:,1])
	roc_auc = auc(fpr, tpr)
	fig = plt.plot(fpr, tpr, label=name + ' (area = %0.3f)' % roc_auc)
	# x = np.arange(0.00001, 1, 0.001)
	# y = 1/x
	# plt.plot(x, y, 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('Higgs Boson Receiver Operating Characteristics')
	plt.legend(loc="lower right")

def plot_show():
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


def array_index_to_matrix_index(index):
	for row in range(NUM_PIXELS):
		if index < (NUM_PIXELS * row) + NUM_PIXELS and index >= NUM_PIXELS * row:
			return index - NUM_PIXELS * row, row
	return -1, -1

if __name__ == '__main__':
	higgs_file_data = read_data(HIGGS_FILE_NAME)
	visualize_plot(higgs_file_data[224], title="Signal colorflow energy image", x_axis="eta", y_axis="phi")
	# not_higgs_file_data = read_data(NOT_HIGGS_FILE_NAME)
	# visualize_plot(not_higgs_file_data[198], title="Background colorflow energy image", x_axis="eta", y_axis="phi")
	# convert_all_to_jpg()

