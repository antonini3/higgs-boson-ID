__author__ = 'lbronner'

from util import *
from constants import *
from sklearn.cross_validation import train_test_split

import theano
import numpy
import theano.tensor as T


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present

    print '... loading data'

    # Load the dataset
    higgs_data = read_data(HIGGS_FILE_NAME)
    not_higgs_data = read_data(NOT_HIGGS_FILE_NAME)

    higgs_data = zip(higgs_data, [1] * len(higgs_data))
    not_higgs_data = zip(not_higgs_data, [0] * len(not_higgs_data))

    all_data = permute_arrays(higgs_data, not_higgs_data)
    x, y = zip(*all_data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    train_set = zip(x_train, y_train)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=30)
    test_set = zip(x_test, y_test)
    valid_set = zip(x_valid, y_valid)

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval