#~/anaconda2/bin/python
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
batch_size   = 20
num_epochs   = 10
MAX_ITER_NUM = int(60000/batch_size*num_epochs)

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
# create a FIFO queue #
#tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, name=None, cancel_op=None)
filename_queue = tf.train.string_input_producer(csv_file_names, shuffle = True, capacity = 100) # 不设定这里的 num_epochs，可以无限循环 # shullfle足够了 #
print ('filename_queue: ', filename_queue)
reader = tf.TextLineReader()
print ('reader: ', reader)
key, value = reader.read(filename_queue)

record_defaults = [[] for i in range(784+10)] # notice : tf.decode_csv return like this format-type #
print ('record_defaults: ', record_defaults)
parse_record_op = tf.decode_csv(value, record_defaults = record_defaults, field_delim=',') # return a list #
#tf.train.batch(tensors, batch_size, num_threads=1, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None)
batch_op = tf.train.batch([parse_record_op[10:], parse_record_op[0:10]], \
										batch_size = batch_size, \
										capacity   = 1000, \
										num_threads= 100,
										allow_smaller_final_batch = False)
#tf.train.shuffle_batch(tensors, batch_size, capacity, min_after_dequeue, num_threads=1, seed=None, enqueue_many=False, shapes=None, allow_smaller_final_batch=False, shared_name=None, name=None)
#batch_op = tf.train.shuffle_batch([parse_record_op[1:], parse_record_op[0]], \
#										batch_size = batch_size, \
#										num_threads= 4
#										capacity   = 1000 + 3 * batch_size, \
#										min_after_dequeue = 1000 , \
#										allow_smaller_final_batch = False)

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
	coord   = tf.train.Coordinator() ## 进程协调器 ##
	threads = tf.train.start_queue_runners(coord=coord) ## is important ##
	for i in range(MAX_ITER_NUM):
		train_x, train_y = sess.run(batch_op)
		#train_xy = sess.run(parse_record_op)
		#train_x  = np.array(train_xy[10:], dtype=np.float32).reshape(1,784)
		#train_y  = np.array(train_xy[0:10],dtype=np.float32).reshape(1,10)
		#print ('train_x.shape: ', train_x.shape, 'train_y.shape: ', train_y.shape)
		#print ('iter : ', i, 'sess.run(shuffle_batch), train_x.shape', train_x.shape, 'train_y.shape', train_y.shape)
		loss, _ = sess.run([cross_entropy, train_op], feed_dict={x: train_x, y_: train_y})
		if i%10 == 0:
			accuracy = sess.run(accuracy_op, feed_dict={x: test_data.images, y_: test_data.labels})
			print ('iter : ', i, 'cross_entropy : ', loss, 'accuracy : ', accuracy)
		
	coord.request_stop()
	coord.join(threads)


