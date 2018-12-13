#!~/anaconda2/bin/python
# -*- coding:utf-8 -*-

from __future__ import print_function
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import *

batch_size = 100
num_epochs = 200

'''
	An more advanced version than pipe-based input.
	notice : package == tensorflow.contrib.data 
'''
'''
	Clasee Dataset
	A Dataset can be used to represent an input pipeline
	as a collection of elements(nested structures of tensors),
	and a "logical plan" of transformations that act on those elements.
'''

# 1). first must creage a Dataset source; 
# if you has TFRecord format input-data on disk, 
#    you can use it : tf.contrib.data.TFRecordDataset
# elif has some tensors in memory
#    you can use it : tf.contrib.data.Dataset.from_tensors()/from_tensor_slices()
# else:
#    tf.contrib.data.TextLineDataset  ========== text-format file
#    tf.contrib.data.FixedLengthRecordDataset == binary-format file
def My_process_func_inPipe(line):
	elem   = tf.string_split(line, delimiter=',')
	image  = np.array(elem[10:], dtype=tf.float32).reshape(1, 784)
	label  = np.array(elem[0:10],dtype=tf.float32).reshape(1, 10)
	return image, label
def My_process_func_outPipe(batch_list):
	arr    = np.array([i.split(',') for i in batch_list])
	labels = np.array(arr[0:, 0:10], dtype=np.float32)
	images = np.array(arr[0:, 10:], dtype=np.float32)
	return images, labels

file_path  = '../data/MNIST_data_trans_onehot/'
file_names = [file_path +str(i) for i in os.listdir(file_path)]
print ('file_names: ', file_names)
dataset    = tf.contrib.data.TextLineDataset(file_names)
dataset    = dataset.skip(1) # 跳过第一行 #
dataset    = dataset.filter(lambda line: tf.not_equal(tf.substr(line, 0,1), '#')) ## 以#开始的行过滤掉 ##
#dataset    = dataset.filter(lambda line: tf.not_equal(line, []))
dataset    = dataset.shuffle(buffer_size = 1000)
dataset    = dataset.batch(batch_size)
#dataset    = dataset.map(My_process_func_inPipe) # 对每一个元素,进行自己定义的处理 # 还不能特别熟练地掌握 #
dataset    = dataset.repeat(num_epochs)
#dataset    =  tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print ('dataset.output_types:', dataset.output_types)
print ('dataset.output_shapes:', dataset.output_shapes)

iterator   = dataset.make_initializable_iterator() #make_one_shot_iterator()#later not support re-init
next_sample= iterator.get_next()

x  = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
W1 = tf.Variable(tf.random_normal([784, 128]))
W2 = tf.Variable(tf.random_normal([128, 10]))
b1 = tf.Variable(tf.zeros([128]))
b2 = tf.Variable(tf.zeros([10]))

y             = tf.nn.softmax(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1)), W2),b2))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
grad_op       = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op      = grad_op.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy_op        = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init_op    = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(iterator.initializer)
	sess.run(init_op)
	for i in range(num_epochs*60000/batch_size):
		try: sample = sess.run(next_sample)
		except tf.errors.OutOfRangeError: break
		train_x, train_y = My_process_func_outPipe(sample)
		loss, _ = sess.run([cross_entropy, train_op], feed_dict={x: train_x, y_: train_y})
		if i%10 == 0:
			accuracy = sess.run(accuracy_op, feed_dict={x: train_x, y_: train_y})
			print ('iter : ', i, 'cross_entropy : ', loss, 'accuracy : ', accuracy)
