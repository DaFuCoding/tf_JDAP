#!/usr/bin/env python
# encoding: utf-8
"""
Train classify network, such as pnet and rnet and onet without auxiliary task
"""
import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import os.path as osp
sys.path.append(osp.join('.'))
from prepare_data import stat_tfrecords
from nets import JDAP_Net
import train_core
import re

import pdb

flags = tf.app.flags
flags.DEFINE_string("gpu_id", "0", "gpu device id.")
flags.DEFINE_string("logdir", "./logdir/pnet/pnet", "Some visual log information.")
flags.DEFINE_string("model_prefix", "./models/pnet/pnet_OHEM/pnet", "Checkpoint and model path")
flags.DEFINE_string("tfrecords_root", "/home/dafu/workspace/FaceDetect/tf_JDAP/tfrecords/pnet", "Train data.")
flags.DEFINE_integer("tfrecords_num", 4, "Sum of tfrecords.")
flags.DEFINE_integer("image_size", 12, "Netwrok input")
flags.DEFINE_integer("frequent", 100, "Show train result in frequent.")
flags.DEFINE_integer("image_sum", 1031327, "Sum of images in train set.")
# Hyper parameter
flags.DEFINE_integer("batch_size", 128, "Batch size in classification task.")
flags.DEFINE_float("lr", 0.01, "Base learning rate.")
flags.DEFINE_float("lr_decay_factor", 0.1, "learning rate decay factor.")
LR_EPOCH = [7, 13]
flags.DEFINE_integer("end_epoch", 16, "How many epoch training.")

########################################
# Hard sample mining and Loss Function #
########################################
flags.DEFINE_boolean("is_ohem", True, "Whether using ohem in face classification.")
flags.DEFINE_float("ohem_ratio", 0.7, "")
flags.DEFINE_string("loss_type", "SF", "SF: SoftmaxWithLoss. FL: Focal loss.")
flags.DEFINE_float("fl_gamma", 2, "")
flags.DEFINE_boolean("fl_balance", False, "Whether using alpha balance positive and negative.")
flags.DEFINE_float("fl_alpha", 0.25, "")
flags.DEFINE_boolean("is_ERC", True, "Using early rejecting classifier.")
flags.DEFINE_float("ERC_thresh", 0.01, "ERC thresh.")
######################
# Optimization Flags #
######################
flags.DEFINE_string("optimizer", "adam", "Optimization strategy.")
flags.DEFINE_float("adam_beta1", 0.9, "")
flags.DEFINE_float("adam_beta2", 0.999, "")
flags.DEFINE_float("adam_epsilon", 1e-8, "")
flags.DEFINE_float("momentum", 0.9, "A `Tensor` or a floating point value.  The momentum.")

# Logging parameter
flags.DEFINE_boolean("is_feature_visual", True, "Output feature map per frequent")
FLAGS = flags.FLAGS

os.environ.setdefault('CUDA_VISIBLE_DEVICES', FLAGS.gpu_id)


def train_net(net_factory, model_prefix, logdir, end_epoch, netSize, tfrecords, frequent=500):
    # Set logging verbosity
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #########################################
        # Get Detect train data from tfrecords. #
        #########################################
        cls_data_label = stat_tfrecords.ReadTFRecord(tfrecords, netSize, 3, 'reg_label', 4)
        image_batch, cls_label_batch, reg_label_batch = \
            tf.train.shuffle_batch([cls_data_label['image'], cls_data_label['cls_label'], cls_data_label['reg_label']],
                                   batch_size=FLAGS.batch_size, capacity=100000, min_after_dequeue=40000, num_threads=16)

        # Network Forward
        if FLAGS.is_ERC:
            cls_prob_op, bbox_pred_op, ERC1_loss_op, ERC2_loss_op, cls_loss_op, bbox_loss_op, end_points = \
                net_factory(image_batch, cls_label_batch, reg_label_batch)
        else:
            cls_prob_op, bbox_pred_op, cls_loss_op, bbox_loss_op, end_points = \
                net_factory(image_batch, cls_label_batch, reg_label_batch)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)

        boundaries = [int(epoch * FLAGS.image_sum / FLAGS.batch_size) for epoch in LR_EPOCH]
        lr_values = [FLAGS.lr * (FLAGS.lr_decay_factor ** x) for x in range(0, len(LR_EPOCH) + 1)]
        lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)

        optimizer = train_core.configure_optimizer(lr_op)
        if FLAGS.is_ERC:
            train_op = optimizer.minimize(ERC1_loss_op+ERC2_loss_op+cls_loss_op+bbox_loss_op, global_step)
        else:
            train_op = optimizer.minimize(train_core.task_add_weight(cls_loss_op, bbox_loss_op), global_step)

        # Save train summary
        tf.summary.scalar('learning_rate', lr_op)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        if FLAGS.is_ERC:
            tf.summary.scalar('loss_ERC1', ERC1_loss_op)
            tf.summary.scalar('loss_ERC2', ERC2_loss_op)
            tf.summary.scalar('loss_last', cls_loss_op)
            tf.summary.scalar('loss_sum', ERC1_loss_op + ERC2_loss_op + cls_loss_op + bbox_loss_op)
        else:
            tf.summary.scalar('loss_sum', cls_loss_op + bbox_loss_op)

        # Save feature map parameters
        if FLAGS.is_feature_visual:
            for feature_name, feature_val in end_points.items():
                tf.summary.histogram(feature_name, feature_val)

        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        if latest_ckpt is not None:
            saver.restore(sess, latest_ckpt)
            start_epoch = int(next(re.finditer("(\d+)(?!.*\d)", latest_ckpt)).group(0))
        else:
            sess.run(tf.global_variables_initializer())
            start_epoch = 1
        tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        n_step_epoch = int(FLAGS.image_sum / FLAGS.batch_size)
        for cur_epoch in range(start_epoch, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []

            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, cls_loss, bbox_loss, lr, summary_str, gb_step = \
                    sess.run([train_op, cls_prob_op, bbox_pred_op, cls_loss_op, bbox_loss_op, lr_op, summary_op, global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)

                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f learning_rate: %4f"
                          % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f " %
                  (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)


def main(_):
    if FLAGS.image_size == 12:
        net_factory = JDAP_Net.JDAP_12Net
    elif FLAGS.image_size == 24:
        if FLAGS.is_ERC:
            net_factory = JDAP_Net.JDAP_24Net_ERC
        else:
            net_factory = JDAP_Net.JDAP_24Net
    elif FLAGS.image_size == 48:
        net_factory = JDAP_Net.JDAP_48Net

    ''' TFRecords input'''
    cls_tfrecords = []
    tfrecords_num = FLAGS.tfrecords_num
    tfrecords_root = FLAGS.tfrecords_root
    for i in range(tfrecords_num):
        print(tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
        cls_tfrecords.append(tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
    print(cls_tfrecords)

    train_net(net_factory=net_factory, model_prefix=FLAGS.model_prefix, logdir=FLAGS.logdir,
              end_epoch=FLAGS.end_epoch, netSize=FLAGS.image_size, tfrecords=cls_tfrecords,
              frequent=FLAGS.frequent)


if __name__ == '__main__':
    tf.app.run()