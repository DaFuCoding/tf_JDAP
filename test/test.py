#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib import slim
inputs = tf.constant(1.0, shape=[1, 3, 3, 1])
weights = tf.constant(1.0, shape=[3, 3, 1, 1])

x3 = tf.constant(1.0, shape=[1, 5, 5, 1])
y2 = tf.nn.conv2d(x3, weights, strides=[1, 2, 2, 1], padding='SAME')

#result = tf.nn.conv2d_transpose(inputs, weights, [1,6,6,1],2)

y3 = tf.nn.conv2d_transpose(y2, filter=weights, output_shape=[1,6,6,1], strides=[1,2,2,1], padding="SAME")
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
print(sess.run(y2))
print(sess.run(y3))