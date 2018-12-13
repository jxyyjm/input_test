import argparse
import sys
import os
import json
import numpy as np
import tensorflow as tf
import time
import gzip
import struct
from tensorflow import metrics
from tensorflow.examples.tutorials.mnist import input_data

class trainData:
  def __init__(self, data_path):
    self.train_data = ''
    self.test_data  = ''
    self.test_x     = ''
    self.test_y     = '' 
    
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    self.train_data = mnist.train
    self.test_data  = mnist.test
    self.test_x     = mnist.test.images
    self.test_y     = mnist.test.labels
    print ('debug# train_x :', self.train_data.images.shape)
    print ('debug# train_y :', self.train_data.labels.shape)
    print ('debug# test_x  :', self.test_x.shape)
    print ('debug# test_y  :', self.test_y.shape)
FLAGS = None

def main(_):
  learning_rate   = FLAGS.learning_rate
  training_epochs = FLAGS.training_epochs
  batch_size      = FLAGS.batch_size
  print ('debug# FLAGS :', FLAGS)
  iterData = trainData(FLAGS.data_path)
  global_step = tf.get_variable('global_step', [],initializer = tf.constant_initializer(0), trainable = False)
  with tf.name_scope('input'):
    x  = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
    tf.set_random_seed(1)
  with tf.name_scope("weights"):
    W1 = tf.Variable(tf.random_normal([784, 128]))
    W2 = tf.Variable(tf.random_normal([128, 10]))
  with tf.name_scope("biases"):
    b1 = tf.Variable(tf.zeros([128]))
    b2 = tf.Variable(tf.zeros([10]))
  with tf.name_scope("softmax"):
    y  = tf.nn.softmax(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1)), W2),b2))
    tf.summary.histogram('activations', y)
  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
  with tf.name_scope('train'):
    grad_op  = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = grad_op.minimize(cross_entropy, global_step=global_step)
  with tf.name_scope('test'):
    y_pred       = tf.nn.softmax(tf.add(tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1)), W2),b2))
    y_pred_label = tf.argmax(y_pred, axis=1)
    y_label      = tf.argmax(y_, axis=1)
    correct_prediction = tf.equal(y_pred_label, y_label)
    accuracy_op  = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  merged = tf.summary.merge_all()
  init_op = tf.global_variables_initializer()
  saver = tf.train.Saver()
  print("Variables initialized ...")
  gpu_options = tf.GPUOptions(allow_growth = True) # Import === set GPU #
  with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True, log_device_placement = True)) as sess:
    sess.run(init_op)
    train_writer = tf.summary.FileWriter("eventLog", sess.graph)      
    start_time   = time.time()
    training_num = iterData.train_data.images.shape[0]
    one_epoch_step = int(training_num/batch_size)
    totalStep      = one_epoch_step * training_epochs
    for step in range(totalStep):
      train_x, train_y = iterData.train_data.next_batch(batch_size)
      _, cost, gstep, summary = sess.run(
                [train_op, cross_entropy, global_step, merged],
                feed_dict={x: train_x, y_: train_y})
      train_writer.add_summary(summary, step)
      elapsed_time = time.time() - start_time
      start_time = time.time()
      if step%100 == 0:
        test_accuracy = sess.run(accuracy_op, feed_dict={x: iterData.test_x, y_: iterData.test_y})
        print("Step: ,", (step), " Epoch: ", (step/one_epoch_step), " Cost: ", cost," Time: ", float(elapsed_time*1000), 'test accuracy : ', test_accuracy)
    print "Train Completed."
    print "saving model..."
    saver.save(sess, FLAGS.save_path+"/model.ckpt")
  print("done")       

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
  parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
  parser.add_argument("--data_path", type=str, default="", help="The path for train file")
  parser.add_argument("--save_path", type=str, default="", help="The save path for model")
  parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate of the train")
  parser.add_argument("--training_epochs",type=int, default=5, help="the epoch of the train")
  parser.add_argument("--batch_size", type=int, default=1200, help="The size of one batch")

  FLAGS, unparsed = parser.parse_known_args()
  print ('debug# after parser.parse_known_args FLAGS:', FLAGS)
  tf.app.run(main=main)
