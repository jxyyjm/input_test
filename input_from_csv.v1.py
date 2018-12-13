#~/anaconda2/bin/python
# -*- coding:utf-8 -*-

from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

'''
	notice : queue-based input API can be cleanly replaced by dataset API 
'''

def Get_files_path(file_path=''):
	csv_file_path  = '../data/MNIST_data_trans_csv_files/'
	#csv_file_path  = file_path
	csv_file_names = os.listdir(csv_file_path)
	csv_file_names = [csv_file_path +i for i in csv_file_names]
	return csv_file_names

csv_file_names = Get_files_path()
# create a FIFO queue #
filename_queue = tf.train.string_input_producer(csv_file_names)
print ('filename_queue: ', filename_queue)
reader = tf.TextLineReader()
print ('reader: ', reader)
key, value = reader.read(filename_queue)

record_defaults = [[] for i in range(785)] # notice : tf.decode_csv return like this format-type #
print ('record_defaults: ', record_defaults)
parse_record_op = tf.decode_csv(value, record_defaults = record_defaults, field_delim=',') # return a list #
print ('parse_record: ', len(parse_record_op))
#feature        = parse_record[1:]
#label          = parse_record[0]

random_par = tf.Variable(tf.random_normal(shape=(2,3), mean=0, stddev=1.0, dtype=tf.float32))
zeros_par  = tf.Variable(tf.zeros(shape=(2,3), dtype=tf.float32))

init_op    = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	print ('random_par: ', sess.run(random_par))
	print ('zeros_par:  ', sess.run(zeros_par))
	coord   = tf.train.Coordinator() ## 进程协调器 ##
	threads = tf.train.start_queue_runners(coord=coord) ## is important ##
	for i in range(10):
		#x, y = sess.run([feature, label])
		print ('sess.run(parse_record): ', sess.run(parse_record_op))

	coord.request_stop()
	coord.join(threads)
