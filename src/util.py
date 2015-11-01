
from constants import *

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def visualize_plot(arr):
    matrix = array_to_numpy_matrix(arr)
    fig = plt.figure()
    plt.imshow(matrix)
    plt.show()

def array_to_matrix(arr):
    return [[arr[i * 25 + j] for j in range(NUM_PIXELS)] for i in range(NUM_PIXELS)]

# specify type because theano will need type=float64
def array_to_numpy_matrix(arr, type=float):
    return np.matrix(array_to_matrix(arr)).astype(type)

# specify type because theano will need type=float64
def read_data(filename, type=float):
    return np.asarray([np.asarray(x.split()).astype(type) for x in open(filename)])

def permute_arrays(arr1, arr2):
    return np.random.permutation(np.concatenate((arr1, arr2)))

def plot_roc(y_test, y_hat):
    fpr, tpr, threshold = roc_curve(y_test, y_hat)
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

# specify type because theano will need type=float64
def get_random_image(higgs=True, type=float):
    filename = NOT_HIGGS_FILE_NAME
    if higgs:
        filename = HIGGS_FILE_NAME
    with open(filename, 'r') as f:
        image = np.asarray(f.readline().split()).astype(float)
    return array_to_numpy_matrix(image, type=type)

if __name__ == '__main__':
    higgs_file_data = read_data(HIGGS_FILE_NAME)
    visualize_plot(higgs_file_data[0])
