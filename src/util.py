
from constants import *

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# import cv2
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
	imageplot = plt.imshow(matrix, cmap=plt.get_cmap('Paired'))
	# plt.hist(imageplot.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

	plt.colorbar()
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

def boolean_y(y):
	length = y.shape[0]
	ret_y = np.zeros((length, NUM_CLASSES))
	ret_y[range(length),y] = 1
	return ret_y

def permute(x, y):
	# np.random.seed(42)
	perm = range(len(x))
	np.random.shuffle(perm)
	return x[perm], y[perm]


def scatter_plot(x, y, dec_boundry=None):
	higgs = [x[i] for i in xrange(len(x)) if y[i] == 1]
	non_higgs = [x[i] for i in xrange(len(x)) if y[i] == 0]
	x_1_higgs, x_2_higgs = zip(*higgs)
	x_1_non_higgs, x_2_non_higgs = zip(*non_higgs)
	plt.scatter(x_1_higgs, x_2_higgs, color='green')
	plt.scatter(x_1_non_higgs, x_2_non_higgs, color='blue')

def read_higgs_data(filename, max_size=None, skip=0):
	l = []
	with open(filename) as f:
		for i, x in enumerate(f):
			x_array = x.split()[skip:]
			if len(x_array) < IMG_DIMENSION:
				x_array.insert(0, 0.)
			
			l.append(np.asarray(x_array, dtype=float))
			
			sys.stdout.write("Progress from {0}: {1} \r".format(filename, i))
			sys.stdout.flush()
			if max_size is not None and i > max_size:
				break
	return np.asarray(l)


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
	if isinstance(y_probs[0], list):
		y_probs = y_probs[:,1]
	fpr, tpr, threshold = roc_curve(y_test, y_probs)
	roc_auc = auc(fpr, tpr)
	print "ROC:", roc_auc
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

def get_auc(y_test, y_probs):
	fpr, tpr, threshold = roc_curve(y_test, y_probs[:,1])
	return auc(fpr, tpr)

def plot_show():
	plt.show()

def our_plot(x, ys, y_labels=None, y_label='Accuracy', x_label='Epochs', title=''):
	plt.figure()
	for i, y in enumerate(ys):
		plt.plot(x, y, label=y_labels[i] if y_labels is not None else '')

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.ylim([0.0, 1.0])
	plt.legend(loc="lower right")
	plt.title(title)
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


def fearture_extraction():

	# not_higgs_file_data = read_data(NOT_HIGGS_FILE_NAME)
	img = cv2.imread('../images/higgs_25_jpg/higgs_174.jpg',0)


	# ORB
	# # Initiate ORB detector
	# orb = cv2.ORB()
	# # find the keypoints with ORB
	# kp = orb.detect(img,None)
	# print(kp)
	# # compute the descriptors with ORB
	# kp, des = orb.compute(img, kp)
	# print(kp)


	# SIFT
	# sift = cv2.SIFT()
	# kp, des = sift.detectAndCompute(img,None)
	# print(kp)


	# STAR
	# # Initiate STAR detector
	# star = cv2.FeatureDetector_create("STAR")
	# # Initiate BRIEF extractor
	# brief = cv2.DescriptorExtractor_create("BRIEF")
	# # find the keypoints with STAR
	# kp = star.detect(img,None)
	# print(kp)
	# # compute the descriptors with BRIEF
	# kp, des = brief.compute(img, kp)
	# print(kp)

	
	# SURF
	# surf = cv2.SURF(400)
	# # Find keypoints and descriptors directly
	# kp, des = surf.detectAndCompute(img,None)
	# print(len(kp))


	# # FAST - visualize
	# higgs_file_data = read_data(HIGGS_FILE_NAME)
	# img = cv2.imread('../images/higgs_25_jpg/not_higgs_174.jpg',0)
	# fast = cv2.FastFeatureDetector()
	# kp = fast.detect(img,None)
	# img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
	# print(img2)
	# # visualize_plot(img2, title="FAST Keypoints", x_axis="eta", y_axis="phi")
	# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	# cv2.imshow('image',img2)
	# cv2.waitKey(0)


	# FAST - create data
	# # Initiate FAST object with default values
	# for i in range(0, 7280):
	# 	filename = '../images/higgs_25_jpg/higgs_' + str(i) + '.jpg'
	# 	img = cv2.imread(filename,0)
	# 	fast = cv2.FastFeatureDetector()
	# 	kp = fast.detect(img,None)
	# 	for i in range(0, 10):
	# 		f = open('../images/higgs_25_fast.txt', 'a')
	# 		if (i<len(kp)):
	# 			f.write('%d' % kp[i].response)
	# 			f.write(' ')
	# 		else:
	# 			f.write('%d' % 0)
	# 			f.write(' ')
	# 	f.write('\n')


	# HISTOGRAM
	# for i in range(0, 10000):
	# 	filename = '../images/higgs_25_jpg/not_higgs_' + str(i) + '.jpg'
	# 	img = cv2.imread(filename,0)
	# 	hist,bins = np.histogram(img.ravel(),256,[0,256])
	# 	np.savetxt('../images/higgs_buffer.txt',hist,delimiter=' ',fmt='%d')
	# 	with open('../images/higgs_buffer.txt', 'r') as temp_file:
	# 		target = open('../images/not_higgs_25_hist.txt', 'a')
	# 		target.write(" ".join(line.strip() for line in temp_file))
	# 		target.write("\n")
	# 		target.truncate()


	# LAPLACIAN
	# for i in range(0, 10000):
	# 	filename = '../images/higgs_100_jpg/not_higgs_' + str(i) + '.jpg'
	# 	img = cv2.imread(filename,0)
	# 	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	# 	np.savetxt('../images/higgs_buffer.txt',laplacian,delimiter=' ',fmt='%d')
	# 	with open('../images/higgs_buffer.txt', 'r') as temp_file:
	# 		target = open('../images/not_higgs_100.txt', 'a')
	# 		target.write(" ".join(line.strip() for line in temp_file))
	# 		target.write("\n")
	# 		target.truncate()

def plot_feature_selection():
	plt.figure()
	x = [1, 3, 5, 10, 15, 20, 30, 50, 100, 200, 300, 500, 625]
	y = [0.852, 0.857, 0.856, 0.861, 0.873, 0.870, 0.874, 0.879, 0.882, 0.872, 0.877, 0.875, 0.873]
	plt.plot(x, y)
	plt.ylim([0.6, 1.0])
	plt.xlim([0.0, 625.0])
	plt.xlabel('Number of Features for Node')
	plt.ylabel('Area Under the ROC Curve')
	# plt.title('Higgs Boson Receiver Operating Characteristics')
	plt.show()

def plot_depth():
	plt.figure()
	x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
	y = [0.838, 0.848, 0.867, 0.866, 0.873, 0.870, 0.877, 0.876, 0.876, 0.874, 0.882, 0.881]
	plt.ylim([0.6, 1.0])
	plt.xlim([0.0, 12])
	plt.plot(x, y, color='green')
	plt.xlabel('Maximum Depth')
	plt.ylabel('Area Under the ROC Curve')
	# plt.title('Higgs Boson Receiver Operating Characteristics')
	plt.show()

def plot_features_removal():
	plt.figure()
	x = [1, 5, 10, 15, 20, 25, 30]
	y_test = [0.636, 0.744, 0.789, 0.815, 0.837, 0.848, 0.850]
	y_train = [0.871, 0.945, 0.971, 0.982, 0.985, 0.989, 0.988]
	plt.xlim([0.0, 30])
	plt.ylim([0.6, 1.0])
	plt.plot(x, y_test, color='green', label='Test')
	plt.plot(x, y_train, color='magenta', label='Train')
	plt.xlabel('Number of Features')
	plt.ylabel('Area Under the ROC Curve')
	plt.legend(loc="lower right")
	# plt.title('Higgs Boson Receiver Operating Characteristics')
	plt.show()


if __name__ == '__main__':
	plot_features_removal()
	# higgs_file_data = read_data_without_pull(FINE_HIGGS_FILE_NAME_W_PULL) # read_data(HIGGS_FILE_NAME)
	# visualize_plot(higgs_file_data[4], title="Signal colorflow energy image", x_axis=r'$\eta$', y_axis=r"$\phi$")
	# not_higgs_file_data = read_data_without_pull(FINE_NOT_HIGGS_FILE_NAME_W_PULL) # read_data(NOT_HIGGS_FILE_NAME)
	# visualize_plot(not_higgs_file_data[2], title="Background colorflow energy image", x_axis=r'$\eta$', y_axis=r"$\phi$")
	# convert_all_to_jpg()

