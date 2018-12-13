#!~/anaconda2/bin/python

# -*- coding:utf-8 -*-

from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf

num_epochs = 10
batch_size = 100

'''
	notice : queue-based input API can be cleanly replaced by dataset API
 
	reference : https://www.tensorflow.org/api_guides/python/reading_data
'''

## give the read-data or other based method ##

def Get_files_path(): # return a [] contain file_names_path #
	csv_file_path  = '../data/MNIST_data_trans_csv_files/'
	#csv_file_path  = file_path
	csv_file_names = os.listdir(csv_file_path)
	csv_file_names = [csv_file_path +i for i in csv_file_names]
	#print ('@1 Get_files_path done')
	return csv_file_names

def Read_my_file_format(filename_queue): # return a [] contain sinlge parse-result #
	reader     = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [[] for i in range(785)]
	parse_record_op = tf.decode_csv(value, record_defaults = record_defaults, field_delim=',')
	#print ('@2 Read_my_file_format done', parse_record_op)
	return parse_record_op 

def Input_pipe_line(batch_size=100, num_epochs=3):
	csv_file_names = Get_files_path()
	filename_queue = tf.train.string_input_producer(csv_file_names, num_epochs = num_epochs, shuffle = True)
	# shuffle is true, string reading will random #
	single_sample  = Read_my_file_format(filename_queue)
	min_after_dequeue = 10000 ## a buffer will randomly sampel means shuffling ##
	capacity          = min_after_dequeue + 3 * batch_size
	x_batch, y_batch  = tf.train.shuffle_batch([single_sample[1:], single_sample[0]], \
												batch_size = batch_size, \
												capacity = capacity, \
												min_after_dequeue = min_after_dequeue, \
												allow_smaller_final_batch = True)
	#print ('@3 Input_pipe_line done', x_batch, y_batch)
	return x_batch, y_batch
