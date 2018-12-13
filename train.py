#!~/anaconda2/bin/python

from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import depend

## give the operation-doing ##

def train_op():
	x, y_ = depend.Input_pipe_line(batch_size=100, num_epochs=3)
#	x  = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
#	y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
	W1 = tf.Variable(tf.random_normal([784, 128]))
	W2 = tf.Variable(tf.random_normal([128, 10]))
	b1 = tf.Variable(tf.zeros([128]))
	b2 = tf.Variable(tf.zeros([10]))

	y             = tf.nn.softmax(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1)), W2),b2))
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	grad_op       = tf.train.GradientDescentOptimizer(learning_rate=0.5)
	train_op      = grad_op.minimize(cross_entropy)
	return train_op

