import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
import unittest

def prelu(inputs):
    with tf.variable_scope("prelu"):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5
        return pos + neg


def JDAP_12Net(inputs):
    with tf.variable_scope("JDAP_12Net") as scope:
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=prelu, biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001), padding='valid'):
            # net = slim.conv2d(inputs, 10, kernel_size=3, scope='conv1')
            # net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
            #
            # print(net.get_shape())
            #
            # net = slim.conv2d(net, 16, kernel_size=3, scope='conv2')
            # net = slim.conv2d(net, 32, kernel_size=3, scope='conv3')
            # print(net.get_shape())

            net = slim.conv2d(inputs, 12, kernel_size=3, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

            print(net.get_shape())

            net = slim.conv2d(net, 24, kernel_size=3, scope='conv2')
            net = slim.conv2d(net, 36, kernel_size=3, scope='conv3')
            print(net.get_shape())

            conv4_1 = slim.conv2d(net, 2, kernel_size=1, scope='conv4_1', activation_fn=None)
            bbox_pred = slim.conv2d(net, 4, kernel_size=1, scope='conv4_2', activation_fn=None)
            cls_prob = tf.nn.softmax(logits=conv4_1)
            return cls_prob, bbox_pred


class FcnDetector(object):
    def __init__(self, net_factory, model_path):
        with tf.Graph().as_default():
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            self.cls_prob, self.bbox_pred = net_factory(image_reshape)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            with tf.device("/cpu:0"):
                conv1 = tf.get_variable("conv1_w", [3, 3, 3, 12], trainable=False)
                conv1_b = tf.get_variable("conv1_b", [12], trainable=False)
                saver = tf.train.Saver({"JDAP_12Net/conv1/weights": conv1,
                                        "JDAP_12Net/conv1/biases": conv1_b})
                saver.restore(self.sess, model_path)
                print(self.sess.run(conv1))
                print(self.sess.run(conv1_b))
                # saver = tf.train.Saver()
                # saver.restore(self.sess, model_path)
                # conv1 = tf.get_variable("conv1", [3,3,3,12], trainable=False)
                # print(conv1)


model_name = '/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_retrain/pnet-12'


class TestModel(unittest.TestCase):
    def test_get_variables_in_checkpoint_file(self):
        reader = pywrap_tensorflow.NewCheckpointReader(model_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for var in var_to_shape_map:
            print(var, var_to_shape_map[var])
        return var_to_shape_map


if __name__ == '__main__':
    unittest.main()
#fcn = FcnDetector(JDAP_12Net, model_name)
