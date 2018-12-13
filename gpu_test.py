
import tensorflow as tf

a = tf.ones(shape=(2,5), dtype=tf.float32)
b = tf.ones(shape=(5,2), dtype=tf.float32)
c = tf.matmul(a, b)


gpu_options = tf.GPUOptions(allow_growth = True)
with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options, \
										allow_soft_placement = True, \
										log_device_placement = True)) as sess:
	print sess.run(c)

'''
# if you want to point GPU by hand #

with tf.device('/gpu:2'):
	a = 
	b = 
	c = 
with tf.Se....
	...


# if you want to use mult GPU #
for d in ['/gpu:'+str(i) for i in range(8)]
	with tf.device(d):
		a = 
		b = 
		c = 
with tf.Se...
	'''

'''
