
from constants import *

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt


def visualize_plot(arr):
	matrix = array_to_numpy_matrix(arr)
	fig = plt.figure()
	plt.imshow(matrix)
	plt.show()


def array_to_matrix(arr):
	return [[arr[i * 25 + j] for j in range(NUM_PIXELS)] for i in range(NUM_PIXELS)]

def array_to_numpy_matrix(arr):
	return np.matrix(array_to_matrix(arr)).astype(float)

def read_data(filename):
	return [x.split() for x in open(filename)]


if __name__ == '__main__':
	higgs_file_data = read_data(NOT_HIGGS_FILE_NAME)
	visualize_plot(higgs_file_data[0])
	