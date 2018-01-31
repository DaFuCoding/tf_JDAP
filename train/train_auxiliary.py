#!/usr/bin/env python
# encoding: utf-8
import os
import sys
from datetime import datetime
import numpy as np
import os.path as osp
sys.path.append(osp.join('.'))
import tensorflow as tf
from prepare_data import stat_tfrecords
from nets.JDAP_Net import JDAP_48Net_Landmark_Pose
from nets.JDAP_Net import JDAP_48Net_Landmark_Pose_Mean_Shape
from nets.JDAP_Net import JDAP_48Net_Landmark_Pose_Dynamic_Shape
from nets.JDAP_Net import JDAP_48Net_Pose
from nets.JDAP_Net import JDAP_48Net_Landmark
from nets.JDAP_Net import JDAP_aNet
import train_core
import re
import hp_config

flags = hp_config.flags
FLAGS = flags.FLAGS
LR_EPOCH = [7, 13]

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def train_landmark_net(model_prefix, logdir, end_epoch, net_size, tfrecords, attrib_tfrecords,
                       frequent=500):
    with tf.Graph().as_default():
        #######################################################
        # Get Attribute and Detect train data from tfrecords. #
        #######################################################
        cls_data_label_dict = stat_tfrecords.ReadTFRecord(tfrecords, net_size, 3)
        cls_image_batch, cls_label_batch, reg_label_batch = \
            tf.train.shuffle_batch([cls_data_label_dict['image'], cls_data_label_dict['cls_label'],
                                    cls_data_label_dict['reg_label']],
                                   batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=10000, num_threads=16,
                                   allow_smaller_final_batch=True)

        attrib_label_dict = stat_tfrecords.ReadTFRecord(
            attrib_tfrecords, net_size, 3, vector_name=['head_pose', 'landmark_label'],
            vector_dim=[3, 68 * 2])
        attrib_image_batch, attrib_cls_label_batch, attrib_bbox_reg_label_batch, \
        attrib_head_pose_batch, attrib_land_reg_batch = tf.train.shuffle_batch(
            [attrib_label_dict['image'], attrib_label_dict['cls_label'], attrib_label_dict['reg_label'],
             attrib_label_dict['head_pose'], attrib_label_dict['landmark_label']],
            batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=10000, num_threads=16,
            allow_smaller_final_batch=True)

        images = tf.concat([cls_image_batch, attrib_image_batch], axis=0)
        cls_labels = tf.concat([cls_label_batch, attrib_cls_label_batch], axis=0)
        bbox_labels = tf.concat([reg_label_batch, attrib_bbox_reg_label_batch], axis=0)

        # Network Forward

        cls_prob_op, bbox_pred_op, landmark_pred_op, \
        cls_loss_op, bbox_loss_op, landmark_loss_op, end_points = \
            JDAP_48Net_Landmark(inputs=images, label=cls_labels, bbox_target=bbox_labels,
                               landmark_target=attrib_land_reg_batch)
        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)

        boundaries = [int(epoch * FLAGS.image_sum / FLAGS.batch_size) for epoch in LR_EPOCH]
        lr_values = [FLAGS.lr * (FLAGS.lr_decay_factor ** x) for x in range(0, len(LR_EPOCH) + 1)]
        lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)

        optimizer = train_core.configure_optimizer(lr_op)
        train_op = optimizer.minimize(
            train_core.task_add_weight(cls_loss_op, bbox_loss_op, landmark_loss_op=landmark_loss_op), global_step)

        #########################################
        # Save train/verify summary.            #
        #########################################
        # Save feature map parameters
        if FLAGS.is_feature_visual:
            for feature_name, feature_val in end_points.items():
                tf.summary.histogram(feature_name, feature_val)

        tf.summary.scalar('learning_rate', lr_op)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        tf.summary.scalar('landmark_reg_loss', landmark_loss_op)
        tf.summary.scalar('cls_bbox_reg_loss_sum', cls_loss_op + bbox_loss_op)

        #########################################
        # Check point or retrieve model         #
        #########################################
        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        coord = tf.train.Coordinator()
        saver = tf.train.Saver()
        if latest_ckpt is not None:
            saver.restore(sess, latest_ckpt)
            start_epoch = int(next(re.finditer("(\d+)(?!.*\d)", latest_ckpt)).group(0))
        else:
            sess.run(tf.global_variables_initializer())
            start_epoch = 1

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        #########################################
        # Main Training/Verify Loop             #
        #########################################
        # allow_smaller_final_batch, avoid discarding data
        n_step_epoch = int(np.ceil(FLAGS.image_sum / FLAGS.batch_size))
        for cur_epoch in range(start_epoch, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []
            landmark_loss_list = []
            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, landmark_pred, \
                cls_loss, bbox_loss, landmark_loss, \
                lr, summary_str, gb_step = sess.run(
                    [train_op, cls_prob_op, bbox_pred_op, landmark_pred_op,
                     cls_loss_op, bbox_loss_op, landmark_loss_op,
                     lr_op, summary_op, global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)
                landmark_loss_list.append(landmark_loss)
                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f "
                          "landmark_loss: %4f learning_rate: %4f"
                          % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                             np.mean(landmark_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f landmark_loss: %4f"
                  % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                     np.mean(landmark_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)
        coord.request_stop()
        coord.join(threads)


def train_pose_net(model_prefix, logdir, end_epoch, net_size, tfrecords, attrib_tfrecords, frequent=500):
    with tf.Graph().as_default():
        #######################################################
        # Get Attribute and Detect train data from tfrecords. #
        #######################################################
        cls_data_label_dict = stat_tfrecords.ReadTFRecord(tfrecords, net_size, 3)
        cls_image_batch, cls_label_batch, reg_label_batch = \
            tf.train.shuffle_batch([cls_data_label_dict['image'], cls_data_label_dict['cls_label'],
                                    cls_data_label_dict['reg_label']],
                                   batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=10000, num_threads=16,
                                   allow_smaller_final_batch=True)

        attrib_label_dict = stat_tfrecords.ReadTFRecord(
            attrib_tfrecords, net_size, 3, vector_name=['head_pose', 'landmark_label'],
            vector_dim=[3, 68 * 2])
        attrib_image_batch, attrib_cls_label_batch, attrib_bbox_reg_label_batch, \
        attrib_head_pose_batch, attrib_land_reg_batch = tf.train.shuffle_batch(
            [attrib_label_dict['image'], attrib_label_dict['cls_label'], attrib_label_dict['reg_label'],
             attrib_label_dict['head_pose'], attrib_label_dict['landmark_label']],
            batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=10000, num_threads=16,
            allow_smaller_final_batch=True)

        images = tf.concat([cls_image_batch, attrib_image_batch], axis=0)
        cls_labels = tf.concat([cls_label_batch, attrib_cls_label_batch], axis=0)
        bbox_labels = tf.concat([reg_label_batch, attrib_bbox_reg_label_batch], axis=0)

        # Network Forward
        cls_prob_op, bbox_pred_op, head_pose_pred_op, \
        cls_loss_op, bbox_loss_op, head_pose_loss_op, end_points = \
            JDAP_48Net_Pose(inputs=images, label=cls_labels, bbox_target=bbox_labels,
                            pose_reg_target=attrib_head_pose_batch)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)

        boundaries = [int(epoch * FLAGS.image_sum / FLAGS.batch_size) for epoch in LR_EPOCH]
        lr_values = [FLAGS.lr * (FLAGS.lr_decay_factor ** x) for x in range(0, len(LR_EPOCH) + 1)]
        lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)

        optimizer = train_core.configure_optimizer(lr_op)
        train_op = optimizer.minimize(
            train_core.task_add_weight(cls_loss_op, bbox_loss_op,
                                       pose_loss_op=head_pose_loss_op), global_step)

        #########################################
        # Save train/verify summary.            #
        #########################################
        # Save feature map parameters
        if FLAGS.is_feature_visual:
            for feature_name, feature_val in end_points.items():
                tf.summary.histogram(feature_name, feature_val)

        tf.summary.scalar('learning_rate', lr_op)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        tf.summary.scalar('head_pose_loss', head_pose_loss_op)
        tf.summary.scalar('cls_bbox_reg_loss_sum', cls_loss_op + bbox_loss_op)

        #########################################
        # Check point or retrieve model         #
        #########################################
        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        coord = tf.train.Coordinator()
        saver = tf.train.Saver()
        if latest_ckpt is not None:
            saver.restore(sess, latest_ckpt)
            start_epoch = int(next(re.finditer("(\d+)(?!.*\d)", latest_ckpt)).group(0))
        else:
            sess.run(tf.global_variables_initializer())
            start_epoch = 1

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        #########################################
        # Main Training/Verify Loop             #
        #########################################
        # allow_smaller_final_batch, avoid discarding data
        n_step_epoch = int(np.ceil(FLAGS.image_sum / FLAGS.batch_size))
        for cur_epoch in range(start_epoch, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []
            pose_reg_loss_list = []
            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, head_pose_pred, cls_loss, bbox_loss, pose_reg_loss, \
                lr, summary_str, gb_step = sess.run(
                    [train_op, cls_prob_op, bbox_pred_op, head_pose_pred_op,
                     cls_loss_op, bbox_loss_op, head_pose_loss_op,
                     lr_op, summary_op, global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)
                pose_reg_loss_list.append(pose_reg_loss)
                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f "
                          "head_pose_loss: %4f learning_rate: %4f"
                          % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                             np.mean(pose_reg_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f head_pose_loss: %4f"
                  % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                     np.mean(pose_reg_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)
        coord.request_stop()
        coord.join(threads)


def train_attribute_net(net_factory, model_prefix, logdir, end_epoch, net_size, tfrecords, attrib_tfrecords,
                        frequent=500):
    with tf.Graph().as_default():
        #######################################################
        # Get Attribute and Detect train data from tfrecords. #
        #######################################################
        cls_data_label_dict = stat_tfrecords.ReadTFRecord(tfrecords, net_size, 3)
        cls_image_batch, cls_label_batch, reg_label_batch = \
            tf.train.shuffle_batch([cls_data_label_dict['image'], cls_data_label_dict['cls_label'],
                                    cls_data_label_dict['reg_label']],
                                   batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=10000, num_threads=16,
                                   allow_smaller_final_batch=True)

        attrib_label_dict = stat_tfrecords.ReadTFRecord(
            attrib_tfrecords, net_size, 3, vector_name=['head_pose', 'landmark_label'],
            vector_dim=[3, 68 * 2])
        attrib_image_batch, attrib_cls_label_batch, attrib_bbox_reg_label_batch, \
        attrib_head_pose_batch, attrib_land_reg_batch = tf.train.shuffle_batch(
            [attrib_label_dict['image'], attrib_label_dict['cls_label'], attrib_label_dict['reg_label'],
             attrib_label_dict['head_pose'], attrib_label_dict['landmark_label']],
            batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=10000, num_threads=16,
            allow_smaller_final_batch=True)

        images = tf.concat([cls_image_batch, attrib_image_batch], axis=0)
        cls_labels = tf.concat([cls_label_batch, attrib_cls_label_batch], axis=0)
        bbox_labels = tf.concat([reg_label_batch, attrib_bbox_reg_label_batch], axis=0)

        # Network Forward
        cls_prob_op, bbox_pred_op, head_pose_pred_op, landmark_pred_op, \
        cls_loss_op, bbox_loss_op, head_pose_loss_op, landmark_loss_op, end_points = \
            net_factory(inputs=images, label=cls_labels, bbox_target=bbox_labels,
                        pose_reg_target=attrib_head_pose_batch, landmark_target=attrib_land_reg_batch)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)

        boundaries = [int(epoch * FLAGS.image_sum / FLAGS.batch_size) for epoch in LR_EPOCH]
        lr_values = [FLAGS.lr * (FLAGS.lr_decay_factor ** x) for x in range(0, len(LR_EPOCH) + 1)]
        lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)

        optimizer = train_core.configure_optimizer(lr_op)
        train_op = optimizer.minimize(
            train_core.task_add_weight(cls_loss_op, bbox_loss_op,
                                       landmark_loss_op=landmark_loss_op,
                                       pose_loss_op=head_pose_loss_op), global_step)

        #########################################
        # Save train/verify summary.            #
        #########################################
        # Save feature map parameters
        if FLAGS.is_feature_visual:
            for feature_name, feature_val in end_points.items():
                tf.summary.histogram(feature_name, feature_val)

        tf.summary.scalar('learning_rate', lr_op)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        tf.summary.scalar('head_pose_loss', head_pose_loss_op)
        tf.summary.scalar('landmark_reg_loss', landmark_loss_op)
        tf.summary.scalar('cls_bbox_reg_loss_sum', cls_loss_op + bbox_loss_op)

        #########################################
        # Check point or retrieve model         #
        #########################################
        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        coord = tf.train.Coordinator()
        saver = tf.train.Saver()
        if latest_ckpt is not None:
            saver.restore(sess, latest_ckpt)
            start_epoch = int(next(re.finditer("(\d+)(?!.*\d)", latest_ckpt)).group(0))
        else:
            sess.run(tf.global_variables_initializer())
            start_epoch = 1

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        #########################################
        # Main Training/Verify Loop             #
        #########################################
        # allow_smaller_final_batch, avoid discarding data
        n_step_epoch = int(np.ceil(FLAGS.image_sum / FLAGS.batch_size))
        for cur_epoch in range(start_epoch, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []
            landmark_loss_list = []
            pose_reg_loss_list = []
            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, head_pose_pred, landmark_pred, \
                cls_loss, bbox_loss, pose_reg_loss, landmark_loss, \
                lr, summary_str, gb_step = sess.run(
                    [train_op, cls_prob_op, bbox_pred_op, head_pose_pred_op, landmark_pred_op,
                     cls_loss_op, bbox_loss_op, head_pose_loss_op, landmark_loss_op,
                     lr_op, summary_op, global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)
                landmark_loss_list.append(landmark_loss)
                pose_reg_loss_list.append(pose_reg_loss)
                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f "
                          "head_pose_loss: %4f landmark_loss: %4f learning_rate: %4f"
                          % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                             np.mean(pose_reg_loss_list), np.mean(landmark_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f head_pose_loss: %4f landmark_loss: %4f"
                  % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                     np.mean(pose_reg_loss_list), np.mean(landmark_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)
        coord.request_stop()
        coord.join(threads)


def main(_):
    global_param_dict = tf.app.flags.FLAGS.__dict__['__flags']
    for k, v in global_param_dict.items():
        print (k, v)
    #net_factory = JDAP_aNet
    #net_factory = JDAP_48Net_Lanmark_Pose
    #net_factory = JDAP_48Net_Landmark_Pose_Mean_Shape
    net_factory = JDAP_48Net_Landmark_Pose_Dynamic_Shape
    ''' TFRecords input'''
    cls_tfrecords = []
    val_tfrecords = []
    landmakr_tfrecords = []
    head_pose_tfrecords = []
    attribute_tfrecords = []

    tfrecords_num = FLAGS.tfrecords_num
    tfrecords_root = FLAGS.tfrecords_root
    # Classification tfrecords
    for i in range(tfrecords_num):
        print(tfrecords_root + "_wop_pnet-%.5d-of-0000%d" % (i, tfrecords_num))
        cls_tfrecords.append(tfrecords_root + "_wop_pnet-%.5d-of-0000%d" % (i, tfrecords_num))
    print(cls_tfrecords)
    # Attribute tfrecords( head_pose and landmark)
    for i in range(tfrecords_num):
        print(tfrecords_root + "_300WLP_68-%.5d-of-0000%d" % (i, tfrecords_num))
        attribute_tfrecords.append(tfrecords_root + "_300WLP_68-%.5d-of-0000%d" % (i, tfrecords_num))

    train_attribute_net(net_factory=net_factory, model_prefix=FLAGS.model_prefix, logdir=FLAGS.logdir,
                        end_epoch=FLAGS.end_epoch, net_size=FLAGS.image_size, tfrecords=cls_tfrecords,
                        attrib_tfrecords=attribute_tfrecords, frequent=FLAGS.frequent)

    # train_landmark_net(model_prefix=FLAGS.model_prefix, logdir=FLAGS.logdir,
    #                    end_epoch=FLAGS.end_epoch, net_size=FLAGS.image_size, tfrecords=cls_tfrecords,
    #                    attrib_tfrecords=attribute_tfrecords, frequent=FLAGS.frequent)
    # train_pose_net(model_prefix=FLAGS.model_prefix, logdir=FLAGS.logdir,
    #                end_epoch=FLAGS.end_epoch, net_size=FLAGS.image_size, tfrecords=cls_tfrecords,
    #                attrib_tfrecords=attribute_tfrecords, frequent=FLAGS.frequent)


if __name__ == '__main__':
    tf.app.run()
