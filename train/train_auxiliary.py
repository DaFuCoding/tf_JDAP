#!/usr/bin/env python
# encoding: utf-8
import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import os.path as osp

sys.path.append(osp.join('..'))
from configs.config import config
from prepare_data import stat_tfrecords
from nets import JDAP_Net
import train_core

os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(config.gpu_id))
FLAGS = tf.app.flags

def train_landmark_net(net_factory, model_prefix, logdir, end_epoch, netSize, tfrecords, landmark_tfrecords,
                       frequent=500):
    with tf.Graph().as_default():
        #########################################
        # Get Detect train data from tfrecords. #
        #########################################
        cls_images, cls_labels, bbox_reg_labels = stat_tfrecords.ReadTFRecord(tfrecords, netSize, 3)
        cls_image_batch, cls_label_batch, bbox_reg_label_batch = tf.train.shuffle_batch(
            [cls_images, cls_labels, bbox_reg_labels], batch_size=config.BATCH_SIZE, capacity=100000,
            min_after_dequeue=40000, num_threads=16)
        land_images, land_cls, land_reg_labels = stat_tfrecords.ReadLandmarkTFRecord(landmark_tfrecords, netSize, 3)
        land_image_batch, land_cls_label_batch, land_reg_label_batch = tf.train.shuffle_batch(
            [land_images, land_cls, land_reg_labels], batch_size=config.LANDMARK_BATCH_SIZE, capacity=100000,
            min_after_dequeue=40000, num_threads=16)

        images = tf.concat([cls_image_batch, land_image_batch], axis=0)
        labels = tf.concat([cls_label_batch, land_cls_label_batch], axis=0)

        # Network Forward
        cls_prob_op, bbox_pred_op, land_pred_op, cls_loss_op, bbox_loss_op, land_loss_op, end_points \
            = net_factory(images, labels, bbox_reg_label_batch, land_reg_label_batch)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = _configure_learning_rate(config.IMAGESUM, global_step)

        boundaries = [int(epoch * config.IMAGESUM / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
        lr_values = [config.BASE_LR * (config.learning_rate_decay_factor ** x) for x in
                     range(0, len(config.LR_EPOCH) + 1)]
        lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
        optimizer = train_core._configure_optimizer(lr_op)

        train_op = optimizer.minimize(train_core.task_add_weight(cls_loss_op, bbox_loss_op, land_loss_op), global_step)
        tf.summary.scalar('learning_rate', lr_op)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        tf.summary.scalar('landmark_reg_loss', land_loss_op)
        tf.summary.scalar('loss_sum', cls_loss_op + bbox_loss_op + land_loss_op)
        if netSize == 48:
            pass
        # tf.summary.histogram('conv1', end_points['JDAP_48Net_Landmark/conv1'])
        # tf.summary.histogram('conv2', end_points['JDAP_48Net_Landmark/conv2'])
        # tf.summary.histogram('conv3', end_points['JDAP_48Net_Landmark/conv3'])
        # tf.summary.histogram('conv4', end_points['JDAP_48Net_Landmark/conv4'])
        # tf.summary.histogram('fc1', end_points['JDAP_48Net_Landmark/fc1'])
        # tf.summary.histogram('softmax', end_points['JDAP_48Net_Landmark/fc2'])
        # tf.summary.histogram('reg_bbox', end_points['JDAP_48Net_Landmark/fc3'])
        # tf.summary.histogram('landmark', end_points['JDAP_48Net_Landmark/landmark'])

        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        n_step_epoch = int(config.IMAGESUM / config.BATCH_SIZE)
        for cur_epoch in range(1, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []
            landmark_loss_list = []
            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, cls_loss, bbox_loss, landmark_loss, lr, summary_str, gb_step = sess.run(
                    [train_op, cls_prob_op, bbox_pred_op, cls_loss_op, bbox_loss_op, land_loss_op, lr_op, summary_op,
                     global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)
                landmark_loss_list.append(landmark_loss)
                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f landmark loss: %4f learning_rate: %4f"
                          % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                             np.mean(landmark_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f landmark loss: %4f"
                  % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                     np.mean(landmark_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)


def train_pose_net(net_factory, model_prefix, logdir, end_epoch, netSize, tfrecords, pose_tfrecords, frequent=500):
    with tf.Graph().as_default():
        #########################################
        # Get Pose and Detect train data from tfrecords. #
        #########################################
        cls_images, cls_labels, bbox_reg_labels = stat_tfrecords.ReadTFRecord(tfrecords, netSize, 3)
        cls_image_batch, cls_label_batch, bbox_reg_label_batch = tf.train.shuffle_batch(
            [cls_images, cls_labels, bbox_reg_labels], batch_size=config.BATCH_SIZE, capacity=100000,
            min_after_dequeue=40000, num_threads=16)
        pose_images, pose_cls_labels, pose_labels = stat_tfrecords.ReadPoseTFRecord(pose_tfrecords, netSize, 3)
        pose_image_batch, pose_cls_label_batch, pose_label_batch = tf.train.shuffle_batch(
            [pose_images, pose_cls_labels, pose_labels], batch_size=config.POSE_BATCH_SIZE, capacity=10000,
            min_after_dequeue=4000, num_threads=16)
        images = tf.concat([cls_image_batch, pose_image_batch], axis=0)
        labels = tf.concat([cls_label_batch, pose_cls_label_batch], axis=0)

        # Network Forward
        cls_prob_op, bbox_pred_op, pose_reg_pred_op, cls_loss_op, bbox_loss_op, pose_reg_loss_op, end_points = \
            net_factory(inputs=images, label=labels, bbox_target=bbox_reg_label_batch, pose_reg_target=pose_label_batch)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = _configure_learning_rate(config.IMAGESUM, global_step)
        boundaries = [int(epoch * config.IMAGESUM / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
        lr_values = [config.BASE_LR * (config.learning_rate_decay_factor ** x) for x in
                     range(0, len(config.LR_EPOCH) + 1)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, lr_values)
        optimizer = train_core._configure_optimizer(learning_rate)

        train_op = optimizer.minimize(
            train_core.task_add_weight(cls_loss_op, bbox_loss_op, landmark_loss_op=None, pose_loss_op=pose_reg_loss_op),
            global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        tf.summary.scalar('pose_reg_loss', pose_reg_loss_op)
        tf.summary.scalar('loss_sum', cls_loss_op + bbox_loss_op + pose_reg_loss_op)
        if netSize == 48:
            tf.summary.histogram('conv1', end_points['JDAP_48Net_Pose/conv1'])
            tf.summary.histogram('conv2', end_points['JDAP_48Net_Pose/conv2'])
            tf.summary.histogram('conv3', end_points['JDAP_48Net_Pose/conv3'])
            tf.summary.histogram('conv4', end_points['JDAP_48Net_Pose/conv4'])
            tf.summary.histogram('fc1', end_points['JDAP_48Net_Pose/fc1'])
            tf.summary.histogram('softmax', end_points['JDAP_48Net_Pose/fc2'])
            tf.summary.histogram('reg_bbox', end_points['JDAP_48Net_Pose/fc3'])
            tf.summary.histogram('pose_reg', end_points['JDAP_48Net_Pose/pose_reg'])

        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        n_step_epoch = int(config.IMAGESUM / config.BATCH_SIZE)
        for cur_epoch in range(1, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []
            pose_reg_loss_list = []
            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, pose_reg_pred, cls_loss, bbox_loss, pose_reg_loss, lr, summary_str, gb_step = sess.run(
                    [train_op, cls_prob_op, bbox_pred_op, pose_reg_pred_op, cls_loss_op, bbox_loss_op, pose_reg_loss_op,
                     learning_rate, summary_op, global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)
                pose_reg_loss_list.append(pose_reg_loss)
                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f pose_reg_loss: %4f learning_rate: %4f"
                          % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                             np.mean(pose_reg_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f pose_reg_loss: %4f"
                  % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                     np.mean(pose_reg_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)


def train_landmark_pose_net(net_factory, model_prefix, logdir, end_epoch, netSize, tfrecords, landmark_tfrecords,
                            pose_tfrecords, frequent=500):
    with tf.Graph().as_default():
        #########################################
        # Get Pose and Detect train data from tfrecords. #
        #########################################
        cls_images, cls_labels, bbox_reg_labels = stat_tfrecords.ReadTFRecord(tfrecords, netSize, 3)
        cls_image_batch, cls_label_batch, bbox_reg_label_batch = tf.train.shuffle_batch(
            [cls_images, cls_labels, bbox_reg_labels], batch_size=config.BATCH_SIZE, capacity=100000,
            min_after_dequeue=40000, num_threads=16)
        pose_images, pose_cls_labels, pose_labels = stat_tfrecords.ReadPoseTFRecord(pose_tfrecords, netSize, 3)
        pose_image_batch, pose_cls_label_batch, pose_label_batch = tf.train.shuffle_batch(
            [pose_images, pose_cls_labels, pose_labels], batch_size=config.POSE_BATCH_SIZE, capacity=10000,
            min_after_dequeue=4000, num_threads=16)
        land_images, land_cls, land_reg_labels = stat_tfrecords.ReadLandmarkTFRecord(landmark_tfrecords, netSize, 3)
        land_image_batch, land_cls_label_batch, land_reg_label_batch = tf.train.shuffle_batch(
            [land_images, land_cls, land_reg_labels], batch_size=config.LANDMARK_BATCH_SIZE, capacity=100000,
            min_after_dequeue=40000, num_threads=16)

        images = tf.concat([cls_image_batch, land_image_batch, pose_image_batch], axis=0)
        labels = tf.concat([cls_label_batch, land_cls_label_batch, pose_cls_label_batch], axis=0)

        # Network Forward
        cls_prob_op, bbox_pred_op, landmark_pred_op, pose_reg_pred_op, cls_loss_op, bbox_loss_op, landmark_loss_op, pose_reg_loss_op, end_points = \
            net_factory(inputs=images, label=labels, bbox_target=bbox_reg_label_batch,
                        landmark_target=land_reg_label_batch, pose_reg_target=pose_label_batch)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = _configure_learning_rate(config.IMAGESUM, global_step)
        boundaries = [int(epoch * config.IMAGESUM / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
        lr_values = [config.BASE_LR * (config.learning_rate_decay_factor ** x) for x in
                     range(0, len(config.LR_EPOCH) + 1)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, lr_values)
        optimizer = train_core._configure_optimizer(learning_rate)

        train_op = optimizer.minimize(
            train_core.task_add_weight(cls_loss_op, bbox_loss_op, landmark_loss_op=landmark_loss_op,
                                       pose_loss_op=pose_reg_loss_op), global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        tf.summary.scalar('pose_reg_loss', pose_reg_loss_op)
        tf.summary.scalar('loss_sum', cls_loss_op + bbox_loss_op + pose_reg_loss_op)
        if netSize == 48:
            pass
        # tf.summary.histogram('conv1', end_points['JDAP_48Net_Landmark_Pose/conv1'])
        # tf.summary.histogram('conv2', end_points['JDAP_48Net_Landmark_Pose/conv2'])
        # tf.summary.histogram('conv3', end_points['JDAP_48Net_Landmark_Pose/conv3'])
        # tf.summary.histogram('conv4', end_points['JDAP_48Net_Landmark_Pose/conv4'])
        # tf.summary.histogram('fc1', end_points['JDAP_48Net_Landmark_Pose/fc1'])
        # tf.summary.histogram('softmax', end_points['JDAP_48Net_Landmark_Pose/fc2'])
        # tf.summary.histogram('reg_bbox', end_points['JDAP_48Net_Landmark_Pose/fc3'])
        # tf.summary.histogram('landmark', end_points['JDAP_48Net_Landmark_Pose/landmark'])
        # tf.summary.histogram('pose_reg', end_points['JDAP_48Net_Landmark_Pose/pose_reg'])

        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        n_step_epoch = int(config.IMAGESUM / config.BATCH_SIZE)
        for cur_epoch in range(1, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []
            landmark_loss_list = []
            pose_reg_loss_list = []
            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, landmark_pred, pose_reg_pred, cls_loss, bbox_loss, landmark_loss, pose_reg_loss, lr, summary_str, gb_step = sess.run(
                    [train_op, cls_prob_op, bbox_pred_op, landmark_pred_op, pose_reg_pred_op, cls_loss_op, bbox_loss_op,
                     landmark_loss_op, pose_reg_loss_op, learning_rate, summary_op, global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)
                landmark_loss_list.append(landmark_loss)
                pose_reg_loss_list.append(pose_reg_loss)
                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print(
                    "%s: Epoch: %d, cls loss: %4f, bbox loss: %4f landmark loss: %4f pose_reg_loss: %4f learning_rate: %4f"
                    % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                       np.mean(landmark_loss_list), np.mean(pose_reg_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f landmark loss: %4f pose_reg_loss: %4f"
                  % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                     np.mean(landmark_loss_list), np.mean(pose_reg_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)


def train_attribute_net(net_factory, model_prefix, logdir, end_epoch, netSize, tfrecords, attrib_tfrecords,
                        frequent=500):
    with tf.Graph().as_default():
        #########################################
        # Get Pose and Detect train data from tfrecords. #
        #########################################
        cls_images, cls_labels, bbox_reg_labels = stat_tfrecords.ReadTFRecord(tfrecords, netSize, 3)
        cls_image_batch, cls_label_batch, bbox_reg_label_batch = tf.train.shuffle_batch(
            [cls_images, cls_labels, bbox_reg_labels], batch_size=FLAGS.batch_size, capacity=20000,
            min_after_dequeue=10000, num_threads=16, allow_smaller_final_batch=True)

        attrib_images, attrib_cls,  attrib_head_pose_labels, attrib_land_reg_labels = stat_tfrecords.ReadTFRecord(
            attrib_tfrecords, netSize, 3, vector_name=['head_pose', 'landmark_reg'], vector_dim=[3, 7*2])
        attrib_image_batch, attrib_cls_label_batch, attrib_reg_label_batch, attrib_head_pose_batch, attrib_land_reg_batch = \
            tf.train.shuffle_batch(
                [attrib_images, attrib_cls, attrib_head_pose_labels, attrib_land_reg_labels],
                batch_size=FLAGS.batch_size, capacity=20000, min_after_dequeue=10000, num_threads=16,
                allow_smaller_final_batch=True)
        images = tf.concat([cls_image_batch, attrib_image_batch], axis=0)
        labels = tf.concat([cls_label_batch, attrib_cls_label_batch], axis=0)

        # Network Forward
        cls_prob_op, bbox_pred_op, landmark_pred_op, pose_reg_pred_op, cls_loss_op, bbox_loss_op, landmark_loss_op, pose_reg_loss_op, end_points = \
            net_factory(inputs=images, label=labels, bbox_target=bbox_reg_label_batch,
                        landmark_target=attrib_land_reg_labels, pose_reg_target=attrib_head_pose_labels)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = _configure_learning_rate(config.IMAGESUM, global_step)
        boundaries = [int(epoch * config.IMAGESUM / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
        lr_values = [config.BASE_LR * (config.learning_rate_decay_factor ** x) for x in
                     range(0, len(config.LR_EPOCH) + 1)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, lr_values)
        optimizer = train_core._configure_optimizer(learning_rate)

        train_op = optimizer.minimize(
            train_core.task_add_weight(cls_loss_op, bbox_loss_op, landmark_loss_op=landmark_loss_op,
                                       pose_loss_op=pose_reg_loss_op), global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('cls_loss', cls_loss_op)
        tf.summary.scalar('bbox_reg_loss', bbox_loss_op)
        tf.summary.scalar('pose_reg_loss', pose_reg_loss_op)
        tf.summary.scalar('loss_sum', cls_loss_op + bbox_loss_op + pose_reg_loss_op)
        if netSize == 48:
            pass

        model_dir = model_prefix.rsplit('/', 1)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Adaptive use gpu memory
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        summary_op = tf.summary.merge_all()

        n_step_epoch = int(config.IMAGESUM / config.BATCH_SIZE)
        for cur_epoch in range(1, end_epoch + 1):
            cls_loss_list = []
            bbox_loss_list = []
            landmark_loss_list = []
            pose_reg_loss_list = []
            for batch_idx in range(n_step_epoch):
                _, cls_pred, bbox_pred, landmark_pred, pose_reg_pred, cls_loss, bbox_loss, landmark_loss, pose_reg_loss, lr, summary_str, gb_step = sess.run(
                    [train_op, cls_prob_op, bbox_pred_op, landmark_pred_op, pose_reg_pred_op, cls_loss_op, bbox_loss_op,
                     landmark_loss_op, pose_reg_loss_op, learning_rate, summary_op, global_step])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)
                landmark_loss_list.append(landmark_loss)
                pose_reg_loss_list.append(pose_reg_loss)
                if not batch_idx % frequent:
                    summary_writer.add_summary(summary_str, gb_step)
                    print(
                    "%s: Epoch: %d, cls loss: %4f, bbox loss: %4f landmark loss: %4f pose_reg_loss: %4f learning_rate: %4f"
                    % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                       np.mean(landmark_loss_list), np.mean(pose_reg_loss_list), lr))
                    sys.stdout.flush()
            print("%s: Epoch: %d, cls loss: %4f, bbox loss: %4f landmark loss: %4f pose_reg_loss: %4f"
                  % (datetime.now(), cur_epoch, np.mean(cls_loss_list), np.mean(bbox_loss_list),
                     np.mean(landmark_loss_list), np.mean(pose_reg_loss_list)))
            saver.save(sess, model_prefix, cur_epoch)

def main():
    logdir = '../logdir/onet/onet_wider_landmark_pose_FL_gamma2'
    model_prefix = '../models/onet/onet_wider_landmark_pose_FL_gamma2/onet'
    netSize = 48

    ''' TFRecords input'''
    tfrecords_root_path = '/home/dafu/workspace/FaceDetect/tf_JDAP/tfrecords'
    cls_tfrecords_root = os.path.join(tfrecords_root_path, 'onet')
    land_tfrecords_root = os.path.join(tfrecords_root_path, 'onet_landmark')
    pose_tfrecords_root = os.path.join(tfrecords_root_path, 'onet_pose')
    cls_tfrecords = []
    landmark_tfrecords = []
    pose_tfrecords = []
    tfrecords_num = 4
    for i in range(tfrecords_num):
        print(cls_tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
        cls_tfrecords.append(cls_tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
    for i in range(tfrecords_num):
        print(land_tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
        landmark_tfrecords.append(land_tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
    for i in range(tfrecords_num):
        print(pose_tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
        pose_tfrecords.append(pose_tfrecords_root + "-%.5d-of-0000%d" % (i, tfrecords_num))
    print(config)

    # train_landmark_net(net_factory=JDAP_Net.JDAP_48Net_Lanmark, model_prefix=model_prefix, logdir=logdir, end_epoch=config.END_EPOCH,
    # 				   netSize=netSize, tfrecords=cls_tfrecords, landmark_tfrecords=landmark_tfrecords, frequent=100)
    # train_pose_net(net_factory=JDAP_Net.JDAP_48Net_Pose, model_prefix=model_prefix, logdir=logdir, end_epoch=config.END_EPOCH,
    # 			   tfrecords=cls_tfrecords, pose_tfrecords=pose_tfrecords, netSize=48, frequent=100)
    train_landmark_pose_net(JDAP_Net.JDAP_48Net_Lanmark_Pose, model_prefix=model_prefix, logdir=logdir,
                            end_epoch=config.END_EPOCH, netSize=netSize, tfrecords=cls_tfrecords,
                            landmark_tfrecords=landmark_tfrecords, pose_tfrecords=pose_tfrecords, frequent=100)


if __name__ == '__main__':
    main()
