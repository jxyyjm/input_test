#!~/anaconda2/bin/python
# -*- coding:utf-8 -*-

from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
	notice : queue-based input API can be cleanly replaced by dataset API 
'''
batch_size   = 50
num_epochs   = 10
MAX_ITER_NUM = int(60000/batch_size*num_epochs)

class ReadFile:
	def __init__(self, file_path):
		self.file_names = os.listdir(file_path)
		self.files_path = [file_path +i for i in self.file_name]
		self.read_done  = dict([(i, 'no') for i in self.files_path])
		self.read_cur   = ''
		self.handle     = ''
	def update_handle(self, new_handle):
		self.handle = new_handle
	def update_read_done(self, read_done_file):
		if self.read_done[read_done_file] == 'yes':
			print ('=== error === this file has been read before, ', read_done_file)
		else:
			self.read_done[read_done_file] = 'yes'
	def update_cur(self, read_ing_file):
		self.read_cur = read_ing_file
	def update_read_state(new_handle, read_done_file, read_ing_file):
		if self.handle != new_handle:
			self.handle = new_handle:
		else: print ('is still ')
	
def Get_files_path(file_path='../MNIST_data_trans_onehot'):
	#csv_file_path  = '../data/MNIST_data_trans_csv_files_small/'
	#csv_file_path  = '../data/MNIST_data_trans_csv_files/'
	csv_file_path  = file_path
	csv_file_names = os.listdir(csv_file_path)
	csv_file_names = [csv_file_path +i for i in csv_file_names]
	return csv_file_names
def get_test_data(file_path):
	mnist = input_data.read_data_sets(file_path, one_hot=True)
	test_ = mnist.test
	return test_

train_file_path= sys.argv[1]
test_file_path = sys.argv[2]
test_data      = get_test_data(test_file_path)
csv_file_names = Get_files_path(train_file_path)


def get_buff_data(handle, buffer_size = 10000):
	text_list = handle.readlines(buffer_size)
	if len(text_list) < buffer_size: flag = False
	else: flag = True
	return text_list, flag, handle
def get_batch_data(csv_file_names, batch_size):
	for csv_file in csv_file_names:
		handle = open(csv_file, 'r')
		buffer_data, flag, handle = get_buff_data(handle, buffer_size=batch_size)
		return buffer_data
		if flag == False:
			


x  = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
W1 = tf.Variable(tf.random_normal([784, 128]))
W2 = tf.Variable(tf.random_normal([128, 10]))
b1 = tf.Variable(tf.zeros([128]))
b2 = tf.Variable(tf.zeros([10]))

y             = tf.nn.softmax(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1)), W2),b2))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
grad_op       = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op      = grad_op.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy_op        = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


init_op    = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	for i in range(MAX_ITER_NUM):
		
		loss, _ = sess.run([cross_entropy, train_op], feed_dict={x: train_x, y_: train_y})
		if i%10 == 0:
			accuracy = sess.run(accuracy_op, feed_dict={x: test_data.images, y_: test_data.labels})
			print ('iter : ', i, 'cross_entropy : ', loss, 'accuracy : ', accuracy)
		
	coord.request_stop()
	coord.join(threads)


