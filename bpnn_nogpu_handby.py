#!/usr/bin/python
# -*- coding:utf-8 -*

## @time       : 2017-06-17
## @author     : yujianmin
## @reference  : http://blog.csdn.net/yujianmin1990/article/details/49935007
## @what-to-do : try to make a any-layer-nn by hand (one-input-layer; any-hidden-layer; one-output-layer)

from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import numpy as np
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data

## =========================================== ##
'''
== input-layer ==  |  == hidden-layer-1 == | == hidden-layer-2 == | == output-layer ==
I[0]-W[0]B[0]-P[0] |  I[1]-W[1]B[1]-P[1]   |   I[2]-W[2]B[2]-P[2] |    I[3]-W[3]B[3]-P[3]
E[0]-DH[0]-D[0]    |  E[1]-DH[1]-D[1]      |   E[2]-DH[2]-D[2]    |    E[3]-DH[3]-D[3]
DW[0]-DB[0]        |  DW[1]-DB[1]          |   DW[2]-DB[2]        |    DW[3]-DB[3]
'''
#  I--input;     W--weight;               P--probabilty
#  E--error;     DH-delt_active_function; D--delt
#  DW--delt_W;   DB--delt_B
## =========================================== ##

class CMyBPNN:
	def __init__(self, hidden_nodes_list=[10, 10], batch_size=100, epoch=100, lr=0.5, file_log='./a'):
		self.train_data = ''
		self.test_data  = ''
		self.model      = ''
		self.W          = []
		self.B          = []
		self.C          = []
		self.middle_res = {}
		self.lr         = lr
		self.file_log   = file_log
		self.epoch      = epoch
		self.batch_size = batch_size
		self.layer_num  = len(hidden_nodes_list)+2
		self.hidden_nodes_list = hidden_nodes_list
		self.middle_res_file   = './middle.res1'
	def __del__(self):
		self.train_data = ''
		self.test_data  = ''
		self.model      = ''
		self.W          = []
		self.B          = []
		self.C          = []
		self.middle_res = {}
		self.lr         = ''
		self.epoch      = ''
		self.batch_size = ''
		self.layer_num  = ''
		self.hidden_nodes_list = []
		self.middle_res_file   = ''

	def read_data(self):
		mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
		self.train_data = mnist.train
		self.test_data  = mnist.test
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))
	def delt_sigmoid(self, x):
		#return -np.exp(-x)/((1+np.exp(-x))**2)
		return self.sigmoid(x) * (1-self.sigmoid(x))
	def get_label_pred(self, pred_mat):
		return np.argmax(pred_mat, axis=1)
	def compute_layer_input(self, layer_num):
		input = self.middle_res['layer_prob'][layer_num-1]
		return np.dot(input, self.W[layer_num-1]) + self.B[layer_num-1]
	def compute_layer_active(self, layer_num):
		return self.sigmoid(self.middle_res['layer_input'][layer_num])
	def compute_layer_delt(self, layer_num):
		return self.delt_sigmoid(self.middle_res['layer_input'][layer_num])
	def compute_diff(self, A, B):
		if len(A) != len(B):
			print ('A.shape:', A.shape, 'B.shape:', B.shape)
			h = open(self.file_log, 'a')
			h.write('A.shape:' + str(A.shape) + 'B.shape:'+ str(B.shape) +'\n')
			h.close()
			return False
		error_mean= np.mean((A-B)**2)
		if error_mean>0:
			print ('(A-B)**2 = ', error_mean)
			h = open(self.file_log, 'a')
			h.wirte('(A-B)**2 = ' + str(error_mean))
			h.close()
			return False
		else: return True
			
	def initial_weight_parameters(self):
		input_node_num  = self.train_data.images.shape[1]
		output_node_num = self.train_data.labels.shape[1]
		## 构造各层W (输入, 输出) node-num-pair 的 list ##
		node_num_pair   = [(input_node_num, self.hidden_nodes_list[0])]
		for i in xrange(len(self.hidden_nodes_list)):
			if i == len(self.hidden_nodes_list)-1: node_num_pair.append((self.hidden_nodes_list[i], output_node_num))
			else: node_num_pair.append((self.hidden_nodes_list[i], self.hidden_nodes_list[i+1]))
		for node_num_i, node_num_o in node_num_pair:
			W = np.reshape(np.array(np.random.normal(0, 0.001, node_num_i*node_num_o)), (node_num_i, node_num_o))
			B = np.array(np.random.normal(0, 0.001, node_num_o))     ## 正向时的 Bias ##
			self.W.append(W)
			self.B.append(B)
		## W,B,C len is self.layer_num-1 ##
	def initial_middle_parameters(self):
		self.middle_res['delt_W'] = [np.zeros_like(i) for i in self.W]  ## 层与层之间的连接权重       #
		self.middle_res['delt_B'] = [np.zeros_like(i) for i in self.B]  ## 前向传播时的偏倚量         #
		self.middle_res['layer_input'] = ['' for i in xrange(self.layer_num) ]   ## 前向传播时的每层输入       #
		self.middle_res['layer_prob']  = ['' for i in xrange(self.layer_num) ]   ## 前向传播时的每层激活概率   #
		self.middle_res['layer_delt']  = ['' for i in xrange(self.layer_num) ]   ## 每层误差回传时，的起始误差 #
		self.middle_res['layer_wucha'] = ['' for i in xrange(self.layer_num) ]   ## 每层由后层传递过来的误差   #
		self.middle_res['layer_active_delt'] = ['' for i in xrange(self.layer_num) ] ## 前向传播的激活后导数   #
	def forward_propagate(self, batch_x):
		## 前向依层，计算 --> output ## 并保存中间结果 ##
		## 0-layer ====== 1-layer; ====== 2-layer; ===== output-layer ##
		## I[0]-W[0]-B[0] I[1]-W[1]-B[1]  I[2]-W[2]-B[2] 
		for layer_num in xrange(self.layer_num):
			if layer_num == 0:
				self.middle_res['layer_input'][layer_num] = batch_x
				self.middle_res['layer_prob'][layer_num]  = batch_x
				self.middle_res['layer_active_delt'][layer_num] = batch_x
			else:
				self.middle_res['layer_input'][layer_num] = self.compute_layer_input(layer_num)
				self.middle_res['layer_prob'][layer_num]  = self.compute_layer_active(layer_num)
				self.middle_res['layer_active_delt'][layer_num]= self.compute_layer_delt(layer_num)
	def compute_output_prob(self, x):
		for layer_num in xrange(self.layer_num):
			if layer_num == 0:
				output = x
			else:
				layer_num = layer_num -1
				output = self.sigmoid(np.dot(output, self.W[layer_num])+self.B[layer_num])
		return output
	def backward_propagate(self, batch_y):
		## 后向依层，计算 --> delt ## 并保存中间结果 ##
		for layer_num in xrange(self.layer_num, 0, -1):
			layer_num = layer_num - 1
			if layer_num == (self.layer_num -1):
				self.middle_res['layer_wucha'][layer_num] = self.middle_res['layer_prob'][layer_num] - batch_y
				self.middle_res['layer_delt'][layer_num]  = \
					self.middle_res['layer_wucha'][layer_num] * self.middle_res['layer_active_delt'][layer_num]
			else:
				self.middle_res['layer_wucha'][layer_num] = \
					np.dot(self.middle_res['layer_delt'][layer_num+1], self.W[layer_num].T)
				self.middle_res['layer_delt'][layer_num]  = \
					self.middle_res['layer_wucha'][layer_num] * self.middle_res['layer_active_delt'][layer_num]
				self.middle_res['delt_W'][layer_num]      = \
					np.dot(self.middle_res['layer_prob'][layer_num].T, self.middle_res['layer_delt'][layer_num+1])/self.batch_size
				self.middle_res['delt_B'][layer_num]      = np.mean(self.middle_res['layer_delt'][layer_num+1], axis=0)
				
	def update_weight(self):
		# for convient, compute from input to output dir #
		for layer_num in xrange(self.layer_num-1):
			self.W[layer_num] -= self.lr * self.middle_res['delt_W'][layer_num]
			self.B[layer_num] -= self.lr * self.middle_res['delt_B'][layer_num]

	def my_bpnn(self):
		self.read_data()
		self.initial_weight_parameters()
		self.initial_middle_parameters()
		iter_num= self.epoch*int(self.train_data.images.shape[0]/self.batch_size)

		for i in xrange(iter_num):
			batch_x, batch_y = self.train_data.next_batch(self.batch_size)
			# 1) compute predict-y
			self.forward_propagate(batch_x)
			# 2) compute delta-each-layer
			self.backward_propagate(batch_y)
			# 3) update  the par-w-B
			self.update_weight()
			# 4) training is doing... 
			if i%100 == 0:
				batch_x_prob = self.middle_res['layer_prob'][self.layer_num-1]
				test_x_prob  = self.compute_output_prob(self.test_data.images)
				batch_acc, batch_confMat = self.compute_accuracy_confusionMat(batch_y, batch_x_prob)
				test_acc,  test_confMat  = self.compute_accuracy_confusionMat(self.test_data.labels, test_x_prob)
				line = 'epoch:' + str(round(i/self.train_data.images.shape[0], 1)) + 'iter:' + str(i) + 'train_batch_accuracy:' + str(batch_acc) + 'test_accuracy' + str(test_acc)
				h = open(self.file_log, 'a')
				h.write(line+'\n')
				h.close()
				

	def save_middle_res(self, string_head):
		handle_w = open(self.middle_res_file, 'aw')
		for k, v in self.middle_res.iteritems():
			handle_w.write(str(string_head)+'\n')
			handle_w.write(str(k)+'\n')
			if isinstance(v, list):
				num = 0
				for i in v:
					try: shape = i.shape
					except: shape = '0'
					handle_w.write('v['+str(num)+'],'+str(shape)+'\n')
					handle_w.write(str(i)+'\n')
					num +=1
		num = 0
		for i in self.W:
			try: shape = i.shape
			except: shape = '0'
			handle_w.write('W['+str(num)+'],'+str(shape)+'\n')
			handle_w.write(str(i)+'\n')
			num +=1
		num = 0
		for i in self.B:
			try: shape = i.shape
			except: shape = '0'
			handle_w.write('B['+str(num)+'],'+str(shape)+'\n')
			handle_w.write(str(i)+'\n')
			num +=1
		handle_w.close()
	def print_para_shape(self):
		for k, v in self.middle_res.iteritems():
			str_save = ''
			if isinstance(v, list):
				num = 0
				for i in v:
					try: shape = i.shape
					except: shape = '0'
					str_save += str(k) +'  v['+str(num)+'],'+str(shape)+'\t|\t'
					num +=1
			else: str_save = str(k)+':'+str(v)
			print (str_save.strip('|\t'))
		num = 0; str_save = ''
		for i in self.W:
			try: shape = i.shape
			except: shape = '0'
			str_save += 'W['+str(num)+'],'+str(shape)+'\t|\t'
			num +=1
		print (str_save.strip('|\t'))
		num = 0; str_save = ''
		for i in self.B:
			try: shape = i.shape
			except: shape = '0'
			str_save += 'B['+str(num)+'],'+str(shape)+'\t|\t'
			num +=1
		print (str_save.strip('|\t'))
		
	def compute_accuracy_confusionMat(self, real_mat, pred_mat):
		pred_label= self.get_label_pred(pred_mat)
		real_label= self.get_label_pred(real_mat)
		accuracy  = metrics.accuracy_score(real_label, pred_label)
		confu_mat = metrics.confusion_matrix(real_label, pred_label, np.unique(real_label))
		return accuracy, confu_mat

if __name__=='__main__':
	file_log = sys.argv[1]
	CTest = CMyBPNN(hidden_nodes_list=[128], batch_size=100, epoch=5, lr=0.5, file_log=file_log)
	CTest.my_bpnn()
	# epoch: 100  iter: 549800 train_batch_accuracy: 0.98 test_accuracy 0.9717 # [100], batch_size=100, lr=0.05 #
	# epoch: 3    iter: 201500 train_batch_accuracy: 1.0  test_accuracy 0.9788 # [100], batch_size=100, lr=0.5  #
	#CTest = CMyBPNN(hidden_nodes_list=[250, 100], batch_size=150, epoch=50, lr=2.0)
	#CTest.my_bpnn() ## 两层以上hidden-layer就不是特别好调了 ##
	# epoch: 50 iter: 54900 train_batch_accuracy: 1.0 test_accuracy 0.9796 # [250, 100], batch_size=50, lr==2.0 #
