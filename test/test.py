#_*_ coding: utf-8 _*_
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

def test_squeeze():
    x = np.array(range(3*2*1*1), dtype=np.float32)
    x = np.reshape(x, [3, 2, 1, 1])
    np_sq_x = np.squeeze(x)
    print(np_sq_x)
    tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    tensor = tf.squeeze(tensor)
    print(tensor.get_shape())
    print(sess.run(tensor))

def test_square():
    pred = tf.convert_to_tensor([1, 0, 1], dtype=tf.float32)
    label = tf.convert_to_tensor([0, 1, 1], dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.square(pred - label))

    print(sess.run(accuracy))

def test_vec_mul_mat():
    # Default using the last dim
    # So TF default data format is 'NHWC'
    vec = tf.convert_to_tensor(np.array(range(3), dtype=np.float32))
    mat = np.reshape(np.array(range(2*2*3)), [2, 2, 3])
    mat_tensor = tf.convert_to_tensor(mat)
    result = mat * vec
    print(sess.run(mat_tensor))
    print(sess.run(result))

from skimage import io
import cv2
from PIL import Image

if __name__ == '__main__':
    #test_squeeze()
    #test_square()
    fname = '/home/dafu/Pictures/test/test.jpg'
    image = Image.open(fname)
    data = image.load()
    pass
    # with tf.Graph().as_default() as g:
    #     with g.name_scope("dafu") as scope:
    #         # sub_g = tf.Graph()
    #         # sub_a = tf.placeholder("int", name='sub_a')
    #
    #         a = tf.placeholder("float", name='a')
    #         b = tf.placeholder("float", name='b')
    #         c = tf.placeholder("float", name='c')
    #         y1 = tf.add(a, b)
    #         y2 = tf.add(y1, c)
    #         writer = tf.summary.FileWriter('./graphs', graph=g)
    #         # tf_config = tf.ConfigProto()
    #         # tf_config.gpu_options.allow_growth = True
    #         # sess = tf.Session(target='', graph=g, config=tf_config)
    # writer.close()



# inputs = tf.constant(1.0, shape=[1, 3, 3, 1])
# weights = tf.constant(1.0, shape=[3, 3, 1, 1])
#
# x3 = tf.constant(1.0, shape=[1, 5, 5, 1])
# y2 = tf.nn.conv2d(x3, weights, strides=[1, 2, 2, 1], padding='SAME')
#
# #result = tf.nn.conv2d_transpose(inputs, weights, [1,6,6,1],2)
#
# y3 = tf.nn.conv2d_transpose(y2, filter=weights, output_shape=[1,6,6,1], strides=[1,2,2,1], padding="SAME")
# print(sess.run(y2))
# print(sess.run(y3))