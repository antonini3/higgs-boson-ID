from os.path import dirname, join, realpath
import logging
import json, random

import numpy as np

logging.basicConfig(format="[%(asctime)s]\t%(msg)s", level=logging.INFO)


class Dataset(object):

    def __init__(self, filename,  data_types = ['image'], split = (.7, .15, .15), dataset_size = None, verbose = False, random_state = True, r_restriction=None):
        """
        Reads data from file and manages it
        Args:
            filename:           data_filename
            data_types:         list of data wanted - 'image', 'pull', 'mass'
            split:              3-tuple of split between train-val-test - must add up to 1
            dataset_size:       size of dataset, or None to use it all
            r_restriction:      mean and {+,-} amount, or None
        """
        self.verbose = verbose
        self.r_mean, self.r_dev = r_restriction if r_restriction is not None else (None, None)

        if random_state:
            random.seed(94)

        self.log('-- DATASET --')
        self.rootpath = dirname(dirname(realpath(__file__)))
        self.datapath = join(self.rootpath, 'data', 'processed')
        
        self._read_data(filename, dataset_size)
        self._split(split)


    def log(self, *args, **kwargs):
        if self.verbose: 
            logging.info(*args, **kwargs)

    
    def _read_data(self, filename, dataset_size):
        self.log('  Reading data from file...')
        self.raw_data = json.load(open(join(self.datapath, '%s.json' % filename)))
        random.shuffle(self.raw_data)

        if self.r_mean is not None:
            self.log('  Removing all elements with delta R that is %.3e away from %.3e...' % (self.r_dev, self.r_mean))
            self.raw_data = [elem for elem in self.raw_data if abs(self.r_mean - elem['delta_r']) < self.r_dev]

        if dataset_size is not None and dataset_size < len(self.raw_data):
            self.raw_data = self.raw_data[:dataset_size]

        for point in self.raw_data:
            point['image'] = np.nan_to_num(np.asarray(point['image']))

        self.log('  Our dataset total size is %d.' % len(self.raw_data))


    def _split(self, split):
        self.log('  Splitting data as: %.2f, %.2f, %.2f' % split)
        if sum(split) != 1:
            raise Exception('Data split does not sum to one.')

        tr, va, te = split
        dataset_size = len(self.raw_data)

        first_split = int(tr * dataset_size)
        second_split = int((tr + va) * dataset_size)

        self.raw_train = self.raw_data[:first_split]
        self.raw_val = self.raw_data[first_split:second_split]
        self.raw_test = self.raw_data[second_split:]

        self.raw_data = None # To clear some memory


    def _single_pad(self, data):
        temp = np.pad(data.reshape((data.shape[0], 25, 25)), pad_width=[(0,0), (3,4), (3,4)], constant_values=0., mode='constant')
        return temp.reshape((data.shape[0], 32 * 32))


    def _pad(self):
        self.x_train = self._single_pad(self.x_train)
        self.x_val = self._single_pad(self.x_val)
        self.x_test = self._single_pad(self.x_test)


    def _zero_center(self):
        train_mean = np.mean(self.train, axis=0)
        self.x_train -= train_mean
        self.x_val -= train_mean
        self.x_test -= train_mean


    def _normalize(self):
        train_max = np.max(np.abs(self.x_train))
        self.x_train /= train_max
        self.x_val /= train_max
        self.x_test /= train_max


    def _booleanize(self):
        
        def boolean_y(y):
            length = y.shape[0]
            ret_y = np.zeros((length, 2))
            ret_y[range(length), y] = 1
            return ret_y

        self.y_train = boolean_y(self.y_train)
        self.y_val = boolean_y(self.y_val)
        self.y_test = boolean_y(self.y_test)


    def _cnn_preprocess(self):
        self._pad()
        self._zero_center()
        self._normalize()
        

    def get_data(self, cnn_preprocessing=False):
        self.x_train = np.asarray([point['image'] for point in self.raw_train])
        self.x_val = np.asarray([point['image'] for point in self.raw_val])
        self.x_test = np.asarray([point['image'] for point in self.raw_test])

        self.y_train = np.asarray([int(point['is_higgs']) for point in self.raw_train])
        self.y_val = np.asarray([int(point['is_higgs']) for point in self.raw_val])
        self.y_test = np.asarray([int(point['is_higgs']) for point in self.raw_test])
        self._booleanize()

        if cnn_preprocessing:
            self._cnn_preprocess()

        return (self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test)


if __name__ == '__main__':
    dataset = Dataset('rotated', verbose = True)
