# encoding: utf-8
"""
JDAP Net framework:
including:
1. Prelu
2. 12Net, 24Net, 48Net and 48LandmarkNet, 48PoseNet , 48LandmarkPoseNet
3. Result OHEM Loss/Focal Loss of Classification and bbox regression and landmark regression and pose regression
"""
# 0.1.0 version main consider 48Net about landmark and pose attribute
__all__ = ['JDAP_12Net', 'JDAP_24Net', 'JDAP_48Net', 'JDAP_48Net_Lanmark',
           'JDAP_48Net_Pose', 'JDAP_48Net_Lanmark_Pose']

import tensorflow as tf
from configs.cfg import config
import tensorflow.contrib.slim as slim
FLAGS = tf.flags.FLAGS
# TF Framework data format default is 'NHWC'
DATA_FORMAT = 'NHWC'


def _prelu(inputs):
    with tf.variable_scope("prelu"):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32, trainable=True)
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5
        return pos + neg


def focal_loss(cls_prob, cls_label_int64, gamma, alpha):
    """
    brief: Using <Focal Loss for Dense Object Detection> putting more focus on hard, misclassified examples
	Args:
		cls_prob: output without activation
		cls_label_int64: classification label
			positive face label is +1
			negative face label is 0
			part face label     is -1
		gamma: (1-pt)**gamma, coefficient of cross entropy with softmax
		alpha: alpha*(1-pt)**gamma,
	Returns: focal loss with gamma
	"""
    label = tf.cast(cls_label_int64, tf.int64)
    valid_inds = tf.where(label >= 0)
    valid_cls_label = tf.gather(label, valid_inds)
    valid_cls_label = tf.reshape(valid_cls_label, (-1, 1))
    valid_cls_prob = tf.gather(cls_prob, valid_inds)
    valid_cls_prob = tf.reshape(valid_cls_prob, (-1, 2))
    probs = tf.nn.softmax(valid_cls_prob)
    one_hot = tf.one_hot(tf.squeeze(valid_cls_label, axis=1), 2)
    focalLoss = -((1 - probs) ** gamma) * tf.log(probs)
    loss = tf.cast(one_hot, dtype=tf.float32) * focalLoss
    if FLAGS.fl_balance:
        loss_pos = loss[:, 1] * alpha
        loss_neg = loss[:, 0] * (1 - alpha)
        loss = loss_pos + loss_neg
    else:
        loss = tf.reduce_sum(loss, axis=1)
    # if FLAGS.cls_ohem:
    #     ones = tf.ones_like(valid_cls_label, dtype=tf.float32)
    #     keep_num = tf.cast(tf.reduce_sum(ones) * FLAGS.ohem_ratio, tf.int32)
    #     _, k_index = tf.nn.top_k(loss, keep_num)
    #     loss = tf.gather(loss, k_index)
    # Only normalization for positive samples
    valid_inds = tf.where(label >= 1)
    normalizer = tf.maximum(tf.shape(valid_inds)[0], 1)
    return tf.reduce_sum(loss) / tf.cast(normalizer, dtype=tf.float32)


def cls_ohem(cls_prob, cls_label_int64):
    """
    Using OHEM in face classification. Ratio is 0.7
        positive face label is +1
        negative face label is 0
        part face label     is -1
    """
    label = tf.cast(cls_label_int64, tf.int64)
    valid_inds = tf.where(label >= 0)
    valid_cls_label = tf.gather(label, valid_inds)
    valid_cls_prob = tf.gather(cls_prob, valid_inds)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_cls_prob, labels=valid_cls_label)
    loss = tf.reshape(loss, (-1,))
    ones = tf.ones_like(valid_cls_label, dtype=tf.float32)
    if FLAGS.is_ohem:
        keep_num = tf.cast(tf.reduce_sum(ones) * FLAGS.ohem_ratio, tf.int32)
    else:
        keep_num = tf.cast(tf.reduce_sum(ones), tf.int32)
    _, k_index = tf.nn.top_k(loss, keep_num)
    loss = tf.gather(loss, k_index)
    return tf.reduce_mean(loss)


def bbox_ohem(bbox_pred, bbox_target, cls_label_int64):
    """
        positive and part sample have bbox regression
        cls_label = -1 or +1
    """
    use_part_sample = True
    zeros = tf.zeros_like(cls_label_int64, dtype=tf.int64)
    ones = tf.ones_like(cls_label_int64, dtype=tf.int64)
    valid_inds_pos = tf.where(tf.equal(cls_label_int64, 1), ones, zeros)
    valid_cls_bool = valid_inds_pos
    if use_part_sample:
        valid_inds_part = tf.where(tf.equal(cls_label_int64, -1), ones, zeros)
        valid_cls_bool += valid_inds_part
    valid_inds = tf.where(valid_cls_bool > 0)
    bbox_pred = tf.gather(bbox_pred, valid_inds)
    bbox_target = tf.gather(bbox_target, valid_inds)
    square_error = tf.reduce_sum(tf.square(bbox_pred - bbox_target), axis=2)
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_pred_target, cls_label_int64):
    """
        landmark flag : -2 or -4 in cls_label_int64
        tf.where get valid index and tf.gather get valid landmark label
    """
    pred_landmark_num = landmark_pred.get_shape()[-1].value
    label_landmark_num = landmark_pred_target.get_shape()[-1].value
    if pred_landmark_num != label_landmark_num:
        # landmark(int): eyes corner, nose, mouth cornet index [36, 39, 42, 45, 30, 48, 54]
        if pred_landmark_num == 7 * 2:
            landmark_index = tf.constant([36, 39, 42, 45, 30, 48, 54], dtype=tf.int32)
        elif pred_landmark_num == 51 * 2:
            landmark_index = tf.constant(range(17, 68), dtype=tf.int32)
        temp_shape = tf.transpose(tf.reshape(landmark_pred_target, [-1, label_landmark_num // 2, 2]),
                                  [1, 2, 0])
        landmark_pred_target = tf.reshape(tf.transpose(tf.gather(temp_shape, landmark_index), [2, 0, 1]),
                                          [-1, pred_landmark_num])

    valid_inds_one = tf.where(tf.equal(cls_label_int64, -2))
    valid_inds_another = tf.where(tf.equal(cls_label_int64, -4))
    valid_inds = tf.concat([valid_inds_one, valid_inds_another], axis=0)
    valid_landmark_pred = tf.gather(landmark_pred, valid_inds)
    # valid_landmark_label = tf.gather(landmark_pred_target, valid_inds)
    valid_landmark_pred = tf.squeeze(valid_landmark_pred, [1])
    square_error = tf.reduce_sum(tf.square(valid_landmark_pred - landmark_pred_target), axis=1)
    # square_error = tf.reduce_sum(tf.square(valid_landmark_pred - landmark_pred_target), axis=1)
    ones = tf.ones_like(valid_inds, dtype=tf.float32)

   # if FLAGS.is_ohem:
    if False:
        keep_num = tf.cast(tf.reduce_sum(ones) * FLAGS.ohem_ratio, tf.int32)
    else:
        keep_num = tf.cast(tf.reduce_sum(ones), tf.int32)
    _, k_index = tf.nn.top_k(square_error, keep_num)
    loss = tf.gather(square_error, k_index)
    return tf.reduce_mean(loss)


def pose_reg_ohem(pose_logits, pose_reg_label, cls_label_int64):
    """
        pose flag : -3 or -4 in cls_label_int64
        tf.where get valid index and tf.gather get valid pose label
    """
    valid_inds_one = tf.where(tf.equal(cls_label_int64, -3))
    valid_inds_another = tf.where(tf.equal(cls_label_int64, -4))
    valid_inds = tf.concat([valid_inds_one, valid_inds_another], axis=0)
    valid_pose_logits = tf.gather(pose_logits, valid_inds)
    valid_pose_logits = tf.squeeze(valid_pose_logits, [1])
    pose_reg_loss = tf.reduce_sum(tf.square(pose_reg_label - valid_pose_logits), axis=1)
    return tf.reduce_mean(pose_reg_loss)


def JDAP_12Net_wop_relu6(inputs, cls_label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_12Net_wop_relu6") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=tf.nn.relu6, biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001), padding='VALID',
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 10, kernel_size=3, stride=2, scope='conv1_s2')
            print(net.get_shape())

            net = slim.conv2d(net, 16, kernel_size=3, scope='conv2')
            net = slim.conv2d(net, 32, kernel_size=3, scope='conv3')
            print(net.get_shape())

            logits = slim.conv2d(net, 2, kernel_size=1, scope='cls_prob', activation_fn=None)
            bbox_pred = slim.conv2d(net, 4, kernel_size=1, scope='bbox_reg', activation_fn=None)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                logits = tf.squeeze(logits, name='squeeze_cls_prob')
                cls_prob = tf.nn.softmax(logits=logits)
                bbox_pred = tf.squeeze(bbox_pred, name='squeeze_bbox_pred')
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(logits, cls_label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(logits, cls_label, FLAGS.fl_gamma, FLAGS.fl_alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, cls_label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                if mode == 'VERIFY':
                    logits = tf.squeeze(logits, name='verify_squeeze_cls_prob')
                    bbox_pred = tf.squeeze(bbox_pred, name='verify_squeeze_bbox_pred')
                cls_prob = tf.nn.softmax(logits=logits)
                return cls_prob, bbox_pred, end_points


def JDAP_12Net_wo_pooling(inputs, cls_label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_12Net_wo_pooling") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001), padding='VALID',
                            outputs_collections=[end_points_collection]):
            channel_scale = 1.0
            net = slim.conv2d(inputs, 10*channel_scale, kernel_size=3, stride=2, scope='conv1_s2')
            print(net.get_shape())

            net = slim.conv2d(net, 16*channel_scale, kernel_size=3, scope='conv2')
            net = slim.conv2d(net, 32*channel_scale, kernel_size=3, scope='conv3')
            print(net.get_shape())

            logits = slim.conv2d(net, 2, kernel_size=1, scope='cls_prob', activation_fn=None)
            bbox_pred = slim.conv2d(net, 4, kernel_size=1, scope='bbox_reg', activation_fn=None)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                logits = tf.squeeze(logits, name='squeeze_cls_prob')
                cls_prob = tf.nn.softmax(logits=logits)
                bbox_pred = tf.squeeze(bbox_pred, name='squeeze_bbox_pred')
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(logits, cls_label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(logits, cls_label, FLAGS.fl_gamma, FLAGS.fl_alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, cls_label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                if mode == 'VERIFY':
                    logits = tf.squeeze(logits, name='verify_squeeze_cls_prob')
                    bbox_pred = tf.squeeze(bbox_pred, name='verify_squeeze_bbox_pred')
                cls_prob = tf.nn.softmax(logits=logits)
                return cls_prob, bbox_pred, end_points


def JDAP_12Net(inputs, cls_label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_12Net") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001), padding='VALID',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.max_pool2d]):
                net = slim.conv2d(inputs, 10, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1', padding='SAME')
                print(net.get_shape())

                net = slim.conv2d(net, 16, kernel_size=3, scope='conv2')
                net = slim.conv2d(net, 32, kernel_size=3, scope='conv3')
                print(net.get_shape())

                # logits = slim.conv2d(net, 2, kernel_size=1, scope='cls_prob', activation_fn=None)
                # bbox_pred = slim.conv2d(net, 4, kernel_size=1, scope='bbox_reg', activation_fn=None)
                # Older model name
                logits = slim.conv2d(net, 2, kernel_size=1, scope='conv4_1', activation_fn=None)
                bbox_pred = slim.conv2d(net, 4, kernel_size=1, scope='conv4_2', activation_fn=None)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if is_training:
                    logits = tf.squeeze(logits, name='squeeze_cls_prob')
                    cls_prob = tf.nn.softmax(logits=logits)
                    bbox_pred = tf.squeeze(bbox_pred, name='squeeze_bbox_pred')
                    if FLAGS.loss_type == 'SF':
                        cls_loss = cls_ohem(logits, cls_label)
                    elif FLAGS.loss_type == 'FL':
                        cls_loss = focal_loss(logits, cls_label, FLAGS.fl_gamma, FLAGS.fl_alpha)
                    bbox_loss = bbox_ohem(bbox_pred, bbox_target, cls_label)
                    return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
                else:
                    if mode == 'VERIFY':
                        logits = tf.squeeze(logits, name='verify_squeeze_cls_prob')
                        bbox_pred = tf.squeeze(bbox_pred, name='verify_squeeze_bbox_pred')
                    cls_prob = tf.nn.softmax(logits=logits)
                    return cls_prob, bbox_pred, end_points


def JDAP_mNet(inputs, label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_mNet") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                inputs = tf.image.resize_bilinear(inputs, tf.constant([18, 18]))
                # input size : 18 x 18 x 3
                # 9 x 9 x 28
                net = slim.conv2d(inputs, 28, kernel_size=3, stride=2, scope='conv1_s2', padding='SAME')
                print(net.get_shape())
                # 4 x 4 x 48
                net = slim.conv2d(net, 48, kernel_size=3, stride=2, scope='conv2_s2', padding='VALID')
                print(net.get_shape())

                # 3 x 3 x 64
                net = slim.conv2d(net, 64, kernel_size=2, scope='conv3', padding='VALID')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, scope='fc1', activation_fn=_prelu)
                print(net.get_shape())

                cls_prob = slim.fully_connected(net, 2, scope='cls_prob')
                bbox_pred = slim.fully_connected(net, 4, scope='bbox_reg')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, end_points


def JDAP_mNet_normal(inputs, label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_mNet_normal") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                inputs = tf.image.resize_bilinear(inputs, tf.constant([18, 18]))
                # input size : 18 x 18 x 3
                net = slim.conv2d(inputs, 28, kernel_size=3,  scope='conv1', padding='VALID')
                print(net.get_shape())
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME')
                # 8 x 8 x 28
                net = slim.conv2d(net, 48, kernel_size=3, scope='conv2', padding='VALID')
                print(net.get_shape())
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2', padding='SAME')
                # 3 x 3 x 48
                net = slim.conv2d(net, 64, kernel_size=1, scope='conv3', padding='VALID')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, scope='fc1', activation_fn=_prelu)
                print(net.get_shape())

                cls_prob = slim.fully_connected(net, 2, scope='cls_prob')
                bbox_pred = slim.fully_connected(net, 4, scope='bbox_reg')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, end_points


def JDAP_24Net_wop(inputs, label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_24Net_wop") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 28, kernel_size=3, stride=2, scope='conv1_s2', padding='VALID')
                print(net.get_shape())

                net = slim.conv2d(net, 48, kernel_size=3, stride=2, scope='conv2_s2', padding='VALID')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3', padding='VALID')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, activation_fn=_prelu, scope='fc1')
                print(net.get_shape())

                cls_prob = slim.fully_connected(net, 2, scope='cls_prob')
                bbox_pred = slim.fully_connected(net, 4, scope='bbox_reg')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if is_training:
                    if FLAGS.loss_type == 'SF':
                        cls_loss = cls_ohem(cls_prob, label)
                    elif FLAGS.loss_type == 'FL':
                        cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                    bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                    return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
                else:
                    cls_prob = tf.nn.softmax(logits=cls_prob)
                    return cls_prob, bbox_pred, end_points


def JDAP_24Net(inputs, label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_24Net") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 28, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME')
                print(net.get_shape())

                net = slim.conv2d(net, 48, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=2, scope='conv3')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, activation_fn=_prelu, scope='fc1')
                print(net.get_shape())
                cls_prob = slim.fully_connected(net, 2, scope='cls_prob')
                bbox_pred = slim.fully_connected(net, 4, scope='bbox_reg')
                # Older model name
                # cls_prob = slim.fully_connected(net, 2, scope='fc2')
                # bbox_pred = slim.fully_connected(net, 4, scope='fc3')
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if is_training:
            if FLAGS.loss_type == 'SF':
                cls_loss = cls_ohem(cls_prob, label)
            elif FLAGS.loss_type == 'FL':
                cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
        else:
            cls_prob = tf.nn.softmax(logits=cls_prob)
            return cls_prob, bbox_pred, end_points


def JDAP_24Net_ERC(inputs, label=None, bbox_target=None, is_training=True):
    with tf.variable_scope("JDAP_24Net_ERC") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='same')
                print(net.get_shape())
                # ERC1
                ERC1 = slim.fully_connected(slim.flatten(net), 24, scope='ERC1')
                ERC1_prob = slim.fully_connected(ERC1, 2, scope='ERC1_prob')
                if is_training:
                    ERC1_loss = cls_ohem(ERC1_prob, label)
                # If keep All positive samples in every ERC, recall is lower.
                # DR1
                easy_prob = tf.nn.softmax(logits=ERC1_prob)
                # if is_training:
                #     except_easy_neg_index = tf.where(
                #         tf.logical_or(easy_prob[:, 1] > FLAGS.ERC_thresh, tf.not_equal(label, 0)))
                #     DR1_index = tf.squeeze(except_easy_neg_index, axis=1)
                # else:
                DR1_index = tf.squeeze(tf.where(easy_prob[:, 1] > FLAGS.ERC_thresh), axis=1)
                net = tf.gather(net, DR1_index)
                if is_training:
                    label = tf.gather(label, DR1_index)
                    bbox_target = tf.gather(bbox_target, DR1_index)
                    reject_easy_neg_num = FLAGS.batch_size - tf.reduce_sum(tf.ones_like(label, dtype=tf.int64))
                    tf.summary.scalar("reject_easy_neg_num", reject_easy_neg_num)

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())
                # ERC2
                ERC2 = slim.fully_connected(slim.flatten(net), 48, scope='ERC2')
                ERC2_prob = slim.fully_connected(ERC2, 2, scope='ERC2_prob')
                if is_training:
                    ERC2_loss = cls_ohem(ERC2_prob, label)
                # DR2
                medium_prob = tf.nn.softmax(logits=ERC2_prob)
                # if is_training:
                #     except_medium_neg_index = tf.where(
                #         tf.logical_or(medium_prob[:, 1] > FLAGS.ERC_thresh, tf.not_equal(label, 0)))
                #     DR2_index = tf.squeeze(except_medium_neg_index, axis=1)
                # else:
                DR2_index = tf.squeeze(tf.where(medium_prob[:, 1] > FLAGS.ERC_thresh), axis=1)
                net = tf.gather(net, DR2_index)
                if is_training:
                    label = tf.gather(label, DR2_index)
                    bbox_target = tf.gather(bbox_target, DR2_index)
                    reject_medium_neg_num = FLAGS.batch_size - tf.reduce_sum(tf.ones_like(label, dtype=tf.int64))
                    tf.summary.scalar("reject_medium_neg_num", reject_medium_neg_num)

                net = slim.conv2d(net, 96, kernel_size=2, scope='conv3')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, activation_fn=_prelu, scope='fc1')
                print(net.get_shape())
                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if is_training:
            if FLAGS.loss_type == 'SF':
                cls_loss = cls_ohem(cls_prob, label)
            elif FLAGS.loss_type == 'FL':
                cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            return cls_prob, bbox_pred, ERC1_loss, ERC2_loss, cls_loss, bbox_loss, end_points
        else:
            cls_prob = tf.nn.softmax(logits=cls_prob)
            # batch_num = int(inputs.get_shape()[0])
            # reserve_mask = tf.Variable(tf.zeros(batch_num, dtype=tf.int32), trainable=False)
            # mask = tf.gather(DR1_index, DR2_index)
            # reserve_mask = tf.scatter_update(reserve_mask, mask, tf.ones_like(mask, dtype=tf.int32))
            return cls_prob, bbox_pred, DR1_index, DR2_index


def JDAP_48Net(inputs, label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_48Net") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, activation_fn=_prelu, scope='fc1')

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, end_points


def JDAP_48Net_GAN_Occlusion(inputs, gan_inputs, coor_inputs, label=None, bbox_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_48Net_GAN_Occlusion") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                #tf.image.crop_and_resize()
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, activation_fn=_prelu, scope='fc1')

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, end_points


def JDAP_48Net_Landmark(inputs, label=None, bbox_target=None, landmark_target=None, is_training=True):
    with tf.variable_scope("JDAP_48Net_Landmark") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='same')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, activation_fn=_prelu, scope='fc1')

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')

                # Add landmark task
                landmark_pred = slim.fully_connected(net, FLAGS.landmark_num * 2, scope='landmark_reg')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                return cls_prob, bbox_pred, landmark_pred, cls_loss, bbox_loss, landmark_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, landmark_pred


def JDAP_48Net_Pose(inputs, label=None, bbox_target=None, pose_reg_target=None, is_training=True):
    with tf.variable_scope("JDAP_48Net_Pose") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='same')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, activation_fn=_prelu, scope='fc1')

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
                pose_reg_pred = slim.fully_connected(net, 3, scope='head_pose')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                pose_reg_loss = pose_reg_ohem(pose_reg_pred, pose_reg_target, label)
                return cls_prob, bbox_pred, pose_reg_pred, cls_loss, bbox_loss, pose_reg_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, pose_reg_pred


def JDAP_48Net_Landmark_Pose(inputs, label=None, bbox_target=None, landmark_target=None, pose_reg_target=None,
                            is_training=True):
    with tf.variable_scope("JDAP_48Net_Landmark_Pose") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='same')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, activation_fn=_prelu, scope='fc1')
                print(net.get_shape())

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
                # Add Landmark and Pose task
                # landmark_pred = slim.fully_connected(net, FLAGS.LANDMARK_NUM, scope='landmark')
                # pose_reg_pred = slim.fully_connected(net, 3, scope='pose_reg')
                landmark_pred = slim.fully_connected(net, FLAGS.landmark_num * 2, scope='landmark_reg')
                pose_reg_pred = slim.fully_connected(net, 3, scope='head_pose')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                pose_reg_loss = pose_reg_ohem(pose_reg_pred, pose_reg_target, label)
                return cls_prob, bbox_pred, pose_reg_pred, landmark_pred, cls_loss, bbox_loss, pose_reg_loss, landmark_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, pose_reg_pred, landmark_pred, end_points


def JDAP_48Net_Landmark_Pose_Mean_Shape(inputs, label=None, bbox_target=None, landmark_target=None, pose_reg_target=None,
                                        is_training=True):
    mean_shape = tf.constant([0.2357, 0.3737, 0.2430, 0.4884, 0.2576, 0.5933, 0.2716, 0.6889, 0.2923, 0.7940,
                              0.3293, 0.8818, 0.3734, 0.9420, 0.4312, 0.9968, 0.5129, 1.0261, 0.5929, 0.9947,
                              0.6467, 0.9411, 0.6865, 0.8800, 0.7196, 0.7918, 0.7384, 0.6862, 0.7511, 0.5899,
                              0.7643, 0.4853, 0.7710, 0.3705, 0.3002, 0.2729, 0.3316, 0.2378, 0.3718, 0.2286,
                              0.4102, 0.2351, 0.4445, 0.2496, 0.5842, 0.2483, 0.6178, 0.2332, 0.6548, 0.2265,
                              0.6938, 0.2363, 0.7224, 0.2710, 0.5156, 0.3724, 0.5168, 0.4469, 0.5180, 0.5185,
                              0.5183, 0.5780, 0.4693, 0.6208, 0.4892, 0.6290, 0.5157, 0.6376, 0.5416, 0.6283,
                              0.5603, 0.6195, 0.3505, 0.3617, 0.3756, 0.3398, 0.4099, 0.3400, 0.4408, 0.3673,
                              0.4123, 0.3812, 0.3763, 0.3822, 0.5829, 0.3670, 0.6150, 0.3388, 0.6499, 0.3390,
                              0.6736, 0.3611, 0.6489, 0.3802, 0.6130, 0.3801, 0.4088, 0.7505, 0.4472, 0.7251,
                              0.4926, 0.7068, 0.5156, 0.7133, 0.5383, 0.7065, 0.5816, 0.7243, 0.6137, 0.7520,
                              0.5782, 0.7993, 0.5471, 0.8243, 0.5134, 0.8300, 0.4801, 0.8246, 0.4487, 0.7992,
                              0.4178, 0.7495, 0.4837, 0.7427, 0.5142, 0.7422, 0.5445, 0.7426, 0.6078, 0.7508,
                              0.5430, 0.7782, 0.5129, 0.7826, 0.4828, 0.7773], dtype=tf.float32)

    with tf.variable_scope("JDAP_48Net_Landmark_Pose_Mean_Shape") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='same')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, activation_fn=_prelu, scope='fc1')
                print(net.get_shape())

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
                # Add Landmark and Pose task
                landmark_pred = slim.fully_connected(net, FLAGS.landmark_num * 2, scope='landmark_reg')
                # pdb.set_trace()
                # repmat_mean_shape = tf.tile(single_mean_shape, [inputs.get_shape()[0]])
                # mean_shapes = tf.reshape(repmat_mean_shape, [-1, FLAGS.landmark_num * 2])
                # mean_shape  + delta(S)
                landmark_pred = tf.add(mean_shape, landmark_pred)
                pose_reg_pred = slim.fully_connected(net, 3, scope='head_pose')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                pose_reg_loss = pose_reg_ohem(pose_reg_pred, pose_reg_target, label)
                return cls_prob, bbox_pred, pose_reg_pred, landmark_pred, cls_loss, bbox_loss, pose_reg_loss, landmark_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, pose_reg_pred, landmark_pred, end_points


def RotationMatrix(angles):
    # get rotation matrix by rotate angle
    phi = angles[:, 0]
    #gamma = angles[:, 1]
    theta = angles[:, 2]
    # angle quantization
    # quantization_pitch_angle = 15 quantization_yaw_angle = 45 quantization_yaw_angle = 25
    convert_scale = tf.constant(180 / 3.1415, dtype=tf.float32)

    #phi = tf.cast((tf.cast((angles[:, 0] * convert_scale) / 15, dtype=tf.int32)) * 15, dtype=tf.float32) / convert_scale
    phi_cos = tf.reshape(tf.cos(phi), [-1, 1]) * tf.constant([0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
    phi_sin = tf.reshape(tf.sin(phi), [-1, 1]) * tf.constant([0, 0, 0, 0, 0, 1, 0, -1, 0], dtype=tf.float32)
    R_x = tf.reshape(phi_cos + phi_sin, [-1, 3, 3]) + tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=tf.float32)

    gamma = tf.cast((tf.cast((angles[:, 1] * convert_scale) / 35, dtype=tf.int32)) * 35, dtype=tf.float32) / convert_scale
    gamma_cos = tf.reshape(tf.cos(gamma), [-1, 1]) * tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
    gamma_sin = tf.reshape(tf.sin(gamma), [-1, 1]) * tf.constant([0, 0, -1, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
    R_y = tf.reshape(gamma_cos + gamma_sin, [-1, 3, 3]) + tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                                                        dtype=tf.float32)
    #theta = tf.cast((tf.cast((angles[:, 2] * convert_scale) / 25, dtype=tf.int32)) * 25, dtype=tf.float32) / convert_scale
    theta_cos = tf.reshape(tf.cos(theta), [-1, 1]) * tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
    theta_sin = tf.reshape(tf.sin(theta), [-1, 1]) * tf.constant([0, 1, 0, -1, 0, 0, 0, 0, 0], dtype=tf.float32)
    R_z = tf.reshape(theta_cos + theta_sin, [-1, 3, 3]) + tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                                                                      dtype=tf.float32)
    R = R_x * R_y * R_z
    return R

import pdb
def JDAP_48Net_Landmark_Pose_Dynamic_Shape(inputs, label=None, bbox_target=None, landmark_target=None, pose_reg_target=None,
                                            is_training=True):
    mean_shape3d = tf.constant([[0.0802, 0.0811, 0.0938, 0.1062, 0.1246, 0.1657, 0.2180, 0.2896, 0.4070, 0.5342,
                                 0.6274, 0.7040, 0.7712, 0.8133, 0.8430, 0.8716, 0.8919, 0.1519, 0.1955, 0.2516,
                                 0.3053, 0.3540, 0.5631, 0.6166, 0.6760, 0.7383, 0.7866, 0.4501, 0.4410, 0.4321,
                                 0.4265, 0.3647, 0.3905, 0.4273, 0.4668, 0.4972, 0.2201, 0.2537, 0.3041, 0.3501,
                                 0.3059, 0.2541, 0.5614, 0.6095, 0.6610, 0.6993, 0.6569, 0.6028, 0.2833, 0.3312,
                                 0.3901, 0.4212, 0.4537, 0.5140, 0.5632, 0.5073, 0.4614, 0.4154, 0.3710, 0.3319,
                                 0.2973, 0.3800, 0.4198, 0.4613, 0.5539, 0.4592, 0.4187, 0.3799],
                                [0.6231, 0.5374, 0.4583, 0.3891, 0.3194, 0.2710, 0.2506, 0.2338, 0.2172, 0.2226,
                                 0.2298, 0.2430, 0.2851, 0.3514, 0.4189, 0.4963, 0.5822, 0.7785, 0.8166, 0.8314,
                                 0.8298, 0.8184, 0.8088, 0.8153, 0.8114, 0.7906, 0.7472, 0.7319, 0.6876, 0.6439,
                                 0.6005, 0.5418, 0.5403, 0.5345, 0.5366, 0.5354, 0.7178, 0.7413, 0.7395, 0.7156,
                                 0.7106, 0.7089, 0.7047, 0.7242, 0.7205, 0.6932, 0.6892, 0.6954, 0.4273, 0.4625,
                                 0.4829, 0.4769, 0.4797, 0.4532, 0.4129, 0.4067, 0.4003, 0.4014, 0.4052, 0.4155,
                                 0.4297, 0.4484, 0.4491, 0.4441, 0.4160, 0.4339, 0.4352, 0.4382],
                                [0.2044, 0.2481, 0.2865, 0.3352, 0.4202, 0.5468, 0.6890, 0.8215, 0.8846, 0.8473,
                                 0.7332, 0.6060, 0.4914, 0.4134, 0.3682, 0.3335, 0.2936, 0.5755, 0.6464, 0.7020,
                                 0.7413, 0.7655, 0.7875, 0.7737, 0.7461, 0.7039, 0.6433, 0.8227, 0.8955, 0.9705,
                                 0.9986, 0.8655, 0.8982, 0.9173, 0.9062, 0.8795, 0.6462, 0.6957, 0.7021, 0.6977,
                                 0.7097, 0.6904, 0.7186, 0.7326, 0.7384, 0.6980, 0.7335, 0.7407, 0.8248, 0.8921,
                                 0.9310, 0.9397, 0.9377, 0.9114, 0.8539, 0.9140, 0.9402, 0.9423, 0.9314, 0.8961,
                                 0.8303, 0.9074, 0.9227, 0.9162, 0.8559, 0.9207, 0.9223, 0.9115]], dtype=tf.float32)

    with tf.variable_scope("JDAP_48Net_Landmark_Pose_Dynamic_Shape") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=_prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='same')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, activation_fn=_prelu, scope='fc1')
                print(net.get_shape())

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
                # Add Landmark and Pose task
                landmark_pred = slim.fully_connected(net, FLAGS.landmark_num * 2, scope='landmark_reg')
                # pdb.set_trace()
                # repmat_mean_shape = tf.tile(single_mean_shape, [inputs.get_shape()[0]])
                # mean_shapes = tf.reshape(repmat_mean_shape, [-1, FLAGS.landmark_num * 2])
                # mean_shape  + delta(S)
                pose_reg_pred = slim.fully_connected(net, 3, scope='head_pose')

                result = RotationMatrix(pose_reg_pred)
                zeros_temp = tf.expand_dims(tf.zeros_like(pose_reg_pred), [2])
                zeros_same_shape = tf.reshape(tf.tile(zeros_temp, [1, 1, FLAGS.landmark_num]), [-1, 3 * FLAGS.landmark_num])
                repmat_mean_shape3d = tf.reshape(tf.reshape(mean_shape3d, [1, -1]) + zeros_same_shape, [-1, 3, FLAGS.landmark_num])
                dynamic_mean_shape = tf.matmul(result, repmat_mean_shape3d)
                landmark_pred = tf.add(tf.reshape(dynamic_mean_shape[:, :2, :], [-1, FLAGS.landmark_num * 2]), landmark_pred)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                pose_reg_loss = pose_reg_ohem(pose_reg_pred, pose_reg_target, label)
                return cls_prob, bbox_pred, pose_reg_pred, landmark_pred, cls_loss, bbox_loss, pose_reg_loss, landmark_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, pose_reg_pred, landmark_pred, end_points

def JDAP_aNet(inputs, label=None, bbox_target=None, landmark_target=None, pose_reg_target=None, is_training=True):
    with tf.variable_scope("JDAP_aNet") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=tf.nn.relu6, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='VALID',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                inputs = tf.image.resize_bilinear(inputs, tf.constant([36, 36]))
                net = slim.conv2d(inputs, 32, kernel_size=3, scope='conv1')
                print(net.get_shape())
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv2')
                print(net.get_shape())
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, scope='conv3')
                print(net.get_shape())
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4', padding='VALID')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, scope='fc1', activation_fn=tf.nn.relu6)

                # Output layer
                cls_prob = slim.fully_connected(net, 2, scope='cls_prob')
                bbox_pred = slim.fully_connected(net, 4, scope='bbox_reg')
                # Add Landmark and Pose task
                landmark_pred = slim.fully_connected(net, FLAGS.landmark_num * 2, scope='landmark_reg')
                head_pose_pred = slim.fully_connected(net, 3, scope='head_pose')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                head_pose_loss = pose_reg_ohem(head_pose_pred, pose_reg_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                return cls_prob, bbox_pred, head_pose_pred, landmark_pred, cls_loss, bbox_loss, \
                       head_pose_loss, landmark_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, head_pose_pred, landmark_pred, end_points


def JDAP_aNet_Cls(inputs, label=None, bbox_target=None, landmark_target=None, pose_reg_target=None, is_training=True, mode='TRAIN'):
    with tf.variable_scope("JDAP_aNet_Cls") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=tf.nn.relu6, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                inputs = tf.image.resize_bilinear(inputs, tf.constant([36, 36]))
                net = slim.conv2d(inputs, 32, kernel_size=3, stride=2, scope='conv1_s2', padding='SAME')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, stride=2, scope='conv2_s2', padding='SAME')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=3, stride=2, scope='conv3_s2', padding='VALID')
                print(net.get_shape())

                net = slim.conv2d(net, 128, kernel_size=2, scope='conv4', padding='VALID')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 256, scope='fc1', activation_fn=tf.nn.relu6)
                # Output layer
                cls_prob = slim.fully_connected(net, 2, scope='cls_prob')
                bbox_pred = slim.fully_connected(net, 4, scope='bbox_reg')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, end_points