from constants import *
from util import *
from analysis import *
from classifier import *
import tensorflow as tf
import numpy as np

class CNNModel(object):

	def __init__(self):
		self.x = tf.placeholder("float", shape=[None, PADDED_IMAGE_DIMENSION])
		self.y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])

		self.keep_prob = tf.placeholder("float")

		self.y_conv = None

		self.loss_function = None
		self.optimizer = None
		self.minimizer = None


	# Weight initialization
	def _weight_variable(self, shape):
		fan_in = shape[0] * shape[1] * shape[2] if len(shape) > 2 else shape[0]
		weight_init = 1. / np.sqrt(fan_in / 2.)
		initial = tf.truncated_normal(shape, stddev=weight_init)
		return tf.Variable(initial)

	# Bias initialization
	def _bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	# Convolution
	def _conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1., 1., 1., 1.], padding='SAME')

	# Max pool
	def _max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def _get_loss_function(self, loss='cross_entropy'):
		if loss == 'cross_entropy':
			self.loss_function = -tf.reduce_sum(self.y_*tf.log(self.y_conv + 1e-7))
			return self.loss_function

	def _get_optimizer(self, optimizer='Adam'):
		if optimizer == 'Adam':
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)
			return self.optimizer

	def _get_minimizer(self, minimizer='accuracy'):
		if minimizer == 'accuracy':
			correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
			self.minimizer = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			return self.minimizer

	def fit(self, X_train, y_train, x_val, y_val, learning_rate=1e-4, batch_size=32, dropout=1.0, max_epochs=100, decay='drop', print_every=50, loss='cross_entropy', optimizer='Adam', minimize='accuracy'):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.dropout = dropout
		self.max_epochs = max_epochs
		self.print_every = print_every
		self.decay = decay

		if type(self.decay) is float:
			decay_type = 'exponential'
		elif type(self.decay) is str and self.decay == 'drop':
			decay_type = decay

		self.sess = tf.InteractiveSession()
		cross_entropy = self._get_loss_function()
		train_step = self._get_optimizer(optimizer)
		minimizer = self._get_minimizer(minimize)

		self.sess.run(tf.initialize_all_variables())

		scores = []
		#iterate over epochs
		for epoch in xrange(max_epochs):
			print "Epoch number: {0}".format(epoch+1)
			num_batches = max(X_train.shape[0] // batch_size, 1)
			x_epoch, y_epoch = permute(X_train, y_train)
			print "  Number of batches: {0}".format(num_batches)
			for i in xrange(num_batches):
				x_batch, y_batch = x_epoch[i * batch_size : (i + 1) * batch_size], y_epoch[i * batch_size : (i + 1) * batch_size]
				assert(len(x_batch) != 0)
				if i > 0 and i % print_every == 0:
					#train_score = self.minimizer.eval(feed_dict={ model.x:x_batch, model.y_: y_batch, model.keep_prob: 1.0})
					train_score = self.score(x_batch, y_batch)
					print("     Step: {0}, Training {1}: {2}".format(i, minimize, train_score))
				train_step.run(feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: self.dropout})
			#val_score = np.mean([self.minimizer.eval(feed_dict={model.x: [x_i], model.y_: [y_i], model.keep_prob: 1.0}) for x_i, y_i in zip(x_val, y_val)])
			val_score = self.score(x_val, y_val)
			print("  Validatin {0}: {1}".format(minimize, val_score))
			scores.append(val_score)
			if decay_type == 'exponential':
				self.learning_rate *= decay
			elif decay_type == 'drop' and epoch > 0:
				average_last_scores = np.mean(scores[epoch], scores[epoch-1])
				print "  Average validation {0} from epoch {1} and {2}: {3}".format(minimize, epoch, epoch-1, average_last_scores)
				score_difference = abs(average_last_scores - val_score)
				print "  difference in {0}: {1}".format(minimize, score_difference)
				if score_difference < 0.008:
					self.learning_rate *= 0.5
					print "  new learning rate: {0}".format(learning_rate)
	
	# not implemented yet. just want forward pass -> check tensorflow documentation
	def predict(self, X):
		predicts = [self.sess.run(self.y_conv, feed_dict={self.x: [x_i], self.keep_prob:self.dropout}) for x_i in X]
		predictions = [pred[0].tolist() for pred in predicts]
		return predictions

	def score(self, X_test, y_test):
		return np.mean([self.minimizer.eval(feed_dict={self.x: [x_i], self.y_: [y_i], self.keep_prob: 1.0}) for x_i, y_i in zip(X_test, y_test)])


class SimpleModel(CNNModel):

	def __init__(self):
		super(SimpleModel, self).__init__()

		# Batchsize, width/height, color channels
		x_image = tf.reshape(self.x, [-1, PADDED_NUM_PIXELS, PADDED_NUM_PIXELS, 1])

		## 5x5 image patches, 1 color channel, 32 outputs
		W_conv1a = self._weight_variable([5, 5, 1, 32])
		b_conv1a = self._bias_variable([32])
		h_conv1a = tf.nn.relu(self._conv2d(x_image, W_conv1a) + b_conv1a)

		h_pool1 = self._max_pool_2x2(h_conv1a)

		# Second layer
		W_conv2a = self._weight_variable([5, 5, 32, 64])
		b_conv2a = self._bias_variable([64])
		h_conv2a = tf.nn.relu(self._conv2d(h_pool1, W_conv2a) + b_conv2a)


		h_pool2 = self._max_pool_2x2(h_conv2a)

		## densely connected
		W_fc1 = self._weight_variable([8 * 8 * 64, 1024])
		b_fc1 = self._bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

		W_fc2 = self._weight_variable([1024, NUM_CLASSES])
		b_fc2 = self._bias_variable([NUM_CLASSES])

		self.y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

class LaNet(CNNModel):

	def __init__(self):
		super(LaNet, self).__init__()

		# Batchsize, width/height, color channels
		x_image = tf.reshape(self.x, [-1, PADDED_NUM_PIXELS, PADDED_NUM_PIXELS, 1])

		# First Layer: (CONV3-32) x 2 - MAXPOOL
		h_conv1a = tf.nn.relu(self._conv2d(x_image, self._weight_variable([3, 3, 1, 32])) + self._bias_variable([32]))
		h_conv1b = tf.nn.relu(self._conv2d(h_conv1a, self._weight_variable([3, 3, 32, 32])) + self._bias_variable([32]))
		h_pool1 = self._max_pool_2x2(h_conv1b)

		# Second Layer: (CONV3-64) x 2 - MAXPOOL
		h_conv2a = tf.nn.relu(self._conv2d(h_pool1, self._weight_variable([3, 3, 32, 64])) + self._bias_variable([64]))
		h_conv2b = tf.nn.relu(self._conv2d(h_conv2a, self._weight_variable([3, 3, 64, 64])) + self._bias_variable([64]))
		h_pool2 = self._max_pool_2x2(h_conv2b)

		# Third Layer: (CONV3-128) x 3 - MAXPOOL
		h_conv3a = tf.nn.relu(self._conv2d(h_pool2, self._weight_variable([3, 3, 64, 128])) + self._bias_variable([128]))
		h_conv3b = tf.nn.relu(self._conv2d(h_conv3a, self._weight_variable([3, 3, 128, 128])) + self._bias_variable([128]))
		h_conv3c = tf.nn.relu(self._conv2d(h_conv3b, self._weight_variable([3, 3, 128, 128])) + self._bias_variable([128]))
		h_pool3 = self._max_pool_2x2(h_conv3c)

		# Fourth Layer: (CONV3-256) x 3 - MAXPOOL
		h_conv4a = tf.nn.relu(self._conv2d(h_pool3, self._weight_variable([3, 3, 128, 256])) + self._bias_variable([256]))
		h_conv4b = tf.nn.relu(self._conv2d(h_conv4a, self._weight_variable([3, 3, 256, 256])) + self._bias_variable([256]))
		h_conv4c = tf.nn.relu(self._conv2d(h_conv4b, self._weight_variable([3, 3, 256, 256])) + self._bias_variable([256]))
		h_pool4 = self._max_pool_2x2(h_conv4c)

		# Fully Connected: 4096 - 4096 - 1000
		h_pool4_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 256])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, self._weight_variable([2 * 2 * 256, 4096])) + self._bias_variable([4096]))
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self._weight_variable([4096, 4096])) + self._bias_variable([4096]))
		h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
		h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, self._weight_variable([4096, 1024])) + self._bias_variable([1024]))
		h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob)

		# Softmax
		self.y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, self._weight_variable([1024, NUM_CLASSES])) + self._bias_variable([NUM_CLASSES]))

class LaNetTwo(CNNModel):

	def __init__(self):
		super(LaNetTwo, self).__init__()

		# Batchsize, width/height, color channels
		x_image = tf.reshape(self.x, [-1, PADDED_NUM_PIXELS, PADDED_NUM_PIXELS, 1])

		# First Layer: (CONV3-32) x 2 - MAXPOOL
		h_conv1a = tf.nn.relu(self._conv2d(x_image, self._weight_variable([3, 3, 1, 32])) + self._bias_variable([32]))
		h_conv1b = tf.nn.relu(self._conv2d(h_conv1a, self._weight_variable([3, 3, 32, 32])) + self._bias_variable([32]))
		h_conv1c = tf.nn.relu(self._conv2d(h_conv1b, self._weight_variable([3, 3, 32, 32])) + self._bias_variable([32]))
		h_pool1 = self._max_pool_2x2(h_conv1c)

		# Second Layer: (CONV3-64) x 2 - MAXPOOL
		h_conv2a = tf.nn.relu(self._conv2d(h_pool1, self._weight_variable([3, 3, 32, 64])) + self._bias_variable([64]))
		h_conv2b = tf.nn.relu(self._conv2d(h_conv2a, self._weight_variable([3, 3, 64, 64])) + self._bias_variable([64]))
		h_conv2c = tf.nn.relu(self._conv2d(h_conv2b, self._weight_variable([3, 3, 64, 64])) + self._bias_variable([64]))
		h_pool2 = self._max_pool_2x2(h_conv2c)

		# Third Layer: (CONV3-128) x 3 - MAXPOOL
		h_conv3a = tf.nn.relu(self._conv2d(h_pool2, self._weight_variable([3, 3, 64, 128])) + self._bias_variable([128]))
		h_conv3b = tf.nn.relu(self._conv2d(h_conv3a, self._weight_variable([3, 3, 128, 128])) + self._bias_variable([128]))
		h_conv3c = tf.nn.relu(self._conv2d(h_conv3b, self._weight_variable([3, 3, 128, 128])) + self._bias_variable([128]))
		h_pool3 = self._max_pool_2x2(h_conv3c)

		# Fourth Layer: (CONV3-256) x 3 - MAXPOOL
		h_conv4a = tf.nn.relu(self._conv2d(h_pool3, self._weight_variable([3, 3, 128, 256])) + self._bias_variable([256]))
		h_conv4b = tf.nn.relu(self._conv2d(h_conv4a, self._weight_variable([3, 3, 256, 256])) + self._bias_variable([256]))
		h_conv4c = tf.nn.relu(self._conv2d(h_conv4b, self._weight_variable([3, 3, 256, 256])) + self._bias_variable([256]))
		h_pool4 = self._max_pool_2x2(h_conv4c)


		# Fifth Layer: (CONV3-512) x 3 - MAXPOOL
		h_conv5a = tf.nn.relu(self._conv2d(h_pool4, self._weight_variable([1, 1, 256, 512])) + self._bias_variable([512]))
		h_conv5b = tf.nn.relu(self._conv2d(h_conv5a, self._weight_variable([1, 1, 512, 512])) + self._bias_variable([512]))
		h_conv5c = tf.nn.relu(self._conv2d(h_conv5b, self._weight_variable([1, 1, 512, 512])) + self._bias_variable([512]))
		h_pool5 = self._max_pool_2x2(h_conv5c)

		# Fully Connected: 4096 - 4096 - 1000
		h_pool5_flat = tf.reshape(h_pool5, [-1, 1 * 1 * 512])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, self._weight_variable([1 * 1 * 512, 4096])) + self._bias_variable([4096]))
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self._weight_variable([4096, 4096])) + self._bias_variable([4096]))
		h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
		h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, self._weight_variable([4096, 1024])) + self._bias_variable([1024]))
		h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob)

		# Softmax
		self.y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, self._weight_variable([1024, NUM_CLASSES])) + self._bias_variable([NUM_CLASSES]))
