#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import tensorflow as tf
from configs.config import config

FLAGS = tf.flags.FLAGS


def configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
    learning_rate: A scalar or `Tensor` learning rate.

    Returns:
    An instance of an optimizer.

    Raises:
    ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=config.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.adam_epsilon)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum, name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', config.optimizer)
    return optimizer


def task_add_weight(cls_loss_op, bbox_loss_op, landmark_loss_op=None, pose_loss_op=None):
    """
        Allocation task weight
    """
    cls_loss_op = cls_loss_op * 1.0
    if landmark_loss_op is not None and pose_loss_op is not None:
        return cls_loss_op + bbox_loss_op * 1.0 + landmark_loss_op * 1.0 + pose_loss_op * 1.0
    elif landmark_loss_op is not None:
        return cls_loss_op + bbox_loss_op * 1.0 + landmark_loss_op * 1.0
    elif pose_loss_op is not None:
        return cls_loss_op + bbox_loss_op * 1.0 + pose_loss_op * 1.0
    return cls_loss_op + bbox_loss_op * 1.0


def compute_accuracy(cls_prob, label, sess):
    # Only compute classify
    keep = (label >= 0)
    tf_cls_prob = tf.convert_to_tensor(np.array(cls_prob))
    tf_cls_prob = tf.reshape(tf_cls_prob, [128, ])
    pred = tf.cast(tf.floor(tf_cls_prob + 0.5), tf.int64)
    pred_vaild = tf.cast(tf.boolean_mask(pred, keep), tf.int64)
    label_vaild = tf.boolean_mask(label, keep)
    print(sess.run(tf.reduce_sum(tf.cast(tf.equal(pred_vaild, label_vaild), tf.float32))))
    print(sess.run(tf.reduce_sum(tf.cast(keep, tf.float32))))
    return tf.reduce_sum(tf.cast(tf.equal(pred_vaild, label_vaild), tf.float32)) / tf.reduce_sum(
        tf.cast(keep, tf.float32))
