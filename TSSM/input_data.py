# coding: utf-8

import numpy as np
import sys
import tensorflow as tf
import scipy.io as sio
from keras.utils.np_utils import to_categorical
import tensorflow_datasets as tfds
import scipy

class DataSubset(object):
	def __init__(self, xs, ys):
		self.xs = xs
		self.n = xs.shape[0]
		self.ys = ys
		self.batch_start = 0
		self.cur_order = np.random.permutation(self.n)

	def next_batch(self, batch_size, reshuffle_after_pass=True, swapaxes=False):
		if self.n < batch_size:
			raise ValueError('Batch size can be at most the dataset size')
		actual_batch_size = min(batch_size, self.n - self.batch_start)
		if actual_batch_size < batch_size:
			if reshuffle_after_pass:
				self.cur_order = np.random.permutation(self.n)
			self.batch_start = 0
		batch_end = self.batch_start + batch_size
		batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
		batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
		self.batch_start += batch_size
		if swapaxes:
			batch_xs = np.swapaxes(batch_xs, 0, 1)
			batch_ys = np.swapaxes(batch_ys, 0, 1)
		return batch_xs, batch_ys

class mnist():

	def __init__(self):
		self.mnist = tf.keras.datasets.mnist		
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train/255 , self.x_test/255

		self.x_train = self.x_train.reshape(60000, 28 * 28)[:55000]
		self.x_train_down_sample = self.x_train.reshape((55000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(55000, 14 * 14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)[:55000]

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000, 14 * 14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)

		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

class fashion_mnist():
	def __init__(self):
		self.mnist = tf.keras.datasets.fashion_mnist		
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

		self.x_train = self.x_train.reshape(60000, 28 * 28)
		self.x_train_down_sample = self.x_train.reshape((60000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(60000, 14 * 14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000, 14 * 14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)
class kmnist():
	def __init__(self):
		train_data =tfds.as_numpy(tfds.load("kmnist", split=tfds.Split.TRAIN, batch_size=-1))
		test_data =tfds.as_numpy(tfds.load("kmnist", split=tfds.Split.TEST, batch_size=-1))
		self.x_train, self.y_train= train_data["image"],train_data["label"]
		self.x_test, self.y_test = test_data["image"], test_data["label"]
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
		self.x_train = self.x_train.reshape(60000, 28*28)
		self.x_train_down_sample = self.x_train.reshape((60000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(
			60000, 14*14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000,  28*28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000,14*14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)
		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)

		train_data =tfds.as_numpy(tfds.load("imagenet_resized", split=tfds.Split.TRAIN, batch_size=-1))
		test_data =tfds.as_numpy(tfds.load("imagenet_resized", split=tfds.Split.TEST, batch_size=-1))
		self.x_train, self.y_train= train_data["image"],train_data["label"]
		self.x_test, self.y_test = test_data["image"], test_data["label"]
		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
		self.x_train = self.x_train.reshape(60000, 28*28)
		self.x_train_down_sample = self.x_train.reshape((60000, 28, 28))
		self.x_train_down_sample = scipy.ndimage.zoom(self.x_train_down_sample, (1, 0.5, 0.5), order=1).reshape(
			60000, 14*14)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000,  28*28)
		self.x_test_down_sample = self.x_test.reshape((10000, 28, 28))
		self.x_test_down_sample = scipy.ndimage.zoom(self.x_test_down_sample, (1, 0.5, 0.5), order=1).reshape(10000,14*14)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)
		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)
		self.train_down_sample = DataSubset(self.x_train_down_sample, self.y_train)
		self.test_down_sample = DataSubset(self.x_test_down_sample, self.y_test)






