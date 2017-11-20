# encoding: utf-8
"""
JDAP Net framework:
including:
1. Prelu
2. 12Net, 24Net, 48Net and 48LandmarkNet, 48PoseNet , 48LandmarkPoseNet
3. Result OHEM Loss/Focal Loss of Classification and bbox regression and landmark regression and pose regression
"""
import tensorflow as tf
from configs.config import config
import tensorflow.contrib.slim as slim

FLAGS = tf.flags.FLAGS


def prelu(inputs):
    with tf.variable_scope("prelu"):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
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
    zeros = tf.zeros_like(cls_label_int64, dtype=tf.int64)
    ones = tf.ones_like(cls_label_int64, dtype=tf.int64)
    valid_inds_part = tf.where(tf.equal(cls_label_int64, -1), ones, zeros)
    #valid_inds_part = tf.where(tf.equal(cls_label_int64, -1), zeros, zeros)
    valid_inds_pos = tf.where(tf.equal(cls_label_int64, 1), ones, zeros)
    valid_cls_bool = valid_inds_part + valid_inds_pos
    valid_inds = tf.where(valid_cls_bool > 0)
    bbox_pred = tf.gather(bbox_pred, valid_inds)
    bbox_target = tf.gather(bbox_target, valid_inds)
    square_error = tf.reduce_sum(tf.square(bbox_pred - bbox_target), axis=2)
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_pred_target, cls_label_int64):
    """
        landmark flag : -2 in cls_label_int64
        tf.where get valid index and tf.gather get valid landmark label
    """
    valid_inds = tf.where(tf.equal(cls_label_int64, -2))
    valid_landmark_pred = tf.gather(landmark_pred, valid_inds)
    # valid_landmark_label = tf.gather(landmark_pred_target, valid_inds)
    valid_landmark_pred = tf.squeeze(valid_landmark_pred, [1])
    square_error = tf.reduce_sum(tf.square(valid_landmark_pred - landmark_pred_target), axis=1)
    # square_error = tf.reduce_sum(tf.square(valid_landmark_pred - landmark_pred_target), axis=1)
    return tf.reduce_mean(square_error)


def pose_reg_ohem(pose_logits, pose_reg_label, cls_label_int64):
    """
		pose flag : -3 in cls_label_int64
		tf.where get valid index and tf.gather get valid pose label
	"""
    valid_inds = tf.where(tf.equal(cls_label_int64, -3))
    valid_pose_logits = tf.gather(pose_logits, valid_inds)
    valid_pose_logits = tf.squeeze(valid_pose_logits, [1])
    pose_reg_loss = tf.reduce_sum(tf.square(pose_reg_label - valid_pose_logits), axis=1)
    return tf.reduce_mean(pose_reg_loss)


def JDAP_wo_pooling_12Net(inputs, cls_label=None, bbox_target=None, is_training=True):
    with tf.variable_scope("JDAP_wo_pooling_12Net") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=tf.nn.relu, biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001), padding='valid',
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(inputs, 10, kernel_size=5, stride=2, padding='SAME', scope='conv1')
            print(net.get_shape())

            net = slim.conv2d(net, 16, kernel_size=3, scope='conv2')
            net = slim.avg_pool2d(net, kernel_size=4, stride=1)

            print(net.get_shape())

            conv4_1 = slim.conv2d(net, 2, kernel_size=1, scope='conv4_1', activation_fn=None)
            bbox_pred = slim.conv2d(net, 4, kernel_size=1, scope='conv4_2', activation_fn=None)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                conv4_1 = tf.squeeze(conv4_1, [1, 2], name='cls_prob')
                cls_prob = tf.nn.softmax(logits=conv4_1)
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(conv4_1, cls_label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(conv4_1, cls_label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, cls_label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=conv4_1)
                return cls_prob, bbox_pred


def JDAP_12Net(inputs, cls_label=None, bbox_target=None, is_training=True):
    with tf.variable_scope("JDAP_12Net") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=prelu, biases_initializer=tf.zeros_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00001), padding='valid',
                            outputs_collections=[end_points_collection]):
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
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                conv4_1 = tf.squeeze(conv4_1, [1, 2], name='cls_prob')
                cls_prob = tf.nn.softmax(logits=conv4_1)
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(conv4_1, cls_label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(conv4_1, cls_label, FLAGS.fl_gamma, FLAGS.fl_alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, cls_label)
                return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=conv4_1)
                return cls_prob, bbox_pred
                #return [cls_prob, bbox_pred, conv4_1]


def JDAP_24Net(inputs, label=None, bbox_target=None, is_training=True):
    with tf.variable_scope("JDAP_24Net") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=prelu, weights_regularizer=slim.l2_regularizer(0.00001),
                            biases_initializer=tf.zeros_initializer(), padding='valid',
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None, biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.00001), activation_fn=None,
                                outputs_collections=[end_points_collection]):
                net = slim.conv2d(inputs, 28, kernel_size=3, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='same')
                print(net.get_shape())

                net = slim.conv2d(net, 48, kernel_size=3, scope='conv2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
                print(net.get_shape())

                net = slim.conv2d(net, 64, kernel_size=2, scope='conv3')
                print(net.get_shape())

                net = slim.flatten(net)
                net = slim.fully_connected(net, 128, activation_fn=prelu, scope='fc1')
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
            return cls_prob, bbox_pred, cls_loss, bbox_loss, end_points
        else:
            cls_prob = tf.nn.softmax(logits=cls_prob)
            return cls_prob, bbox_pred


def JDAP_24Net_ERC(inputs, label=None, bbox_target=None, is_training=True):
    with tf.variable_scope("JDAP_24Net_ERC") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=prelu, weights_regularizer=slim.l2_regularizer(0.00001),
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
                net = slim.fully_connected(net, 128, activation_fn=prelu, scope='fc1')
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


def JDAP_48Net(inputs, label=None, bbox_target=None, is_training=True):
    with tf.variable_scope("JDAP_48Net") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=prelu, weights_regularizer=slim.l2_regularizer(0.00001),
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
                net = slim.fully_connected(net, 256, activation_fn=prelu, scope='fc1')

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
                return cls_prob, bbox_pred


def JDAP_48Net_Lanmark(inputs, label=None, bbox_target=None, landmark_target=None, is_training=True):
    with tf.variable_scope("JDAP_48Net_Landmark") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=prelu, weights_regularizer=slim.l2_regularizer(0.00001),
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
                net = slim.fully_connected(net, 256, activation_fn=prelu, scope='fc1')

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')

                # Add landmark task
                landmark_pred = slim.fully_connected(net, 10, scope='landmark')

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
                            activation_fn=prelu, weights_regularizer=slim.l2_regularizer(0.00001),
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
                net = slim.fully_connected(net, 256, activation_fn=prelu, scope='fc1')

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
                pose_reg_pred = slim.fully_connected(net, 3, scope='pose_reg')

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


def JDAP_48Net_Lanmark_Pose(inputs, label=None, bbox_target=None, landmark_target=None, pose_reg_target=None,
                            is_training=True):
    with tf.variable_scope("JDAP_48Net_Landmark_Pose") as scope:
        end_points_collection = scope.name + '_end_points'
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, weights_initializer=slim.xavier_initializer(),
                            activation_fn=prelu, weights_regularizer=slim.l2_regularizer(0.00001),
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
                net = slim.fully_connected(net, 256, activation_fn=prelu, scope='fc1')

                cls_prob = slim.fully_connected(net, 2, scope='fc2')
                bbox_pred = slim.fully_connected(net, 4, scope='fc3')
                # Add Landmark and Pose task
                landmark_pred = slim.fully_connected(net, 10, scope='landmark')
                pose_reg_pred = slim.fully_connected(net, 3, scope='pose_reg')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if is_training:
                if FLAGS.loss_type == 'SF':
                    cls_loss = cls_ohem(cls_prob, label)
                elif FLAGS.loss_type == 'FL':
                    cls_loss = focal_loss(cls_prob, label, config.FL.gamma, config.FL.alpha)
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
                pose_reg_loss = pose_reg_ohem(pose_reg_pred, pose_reg_target, label)
                return cls_prob, bbox_pred, landmark_pred, pose_reg_pred, cls_loss, bbox_loss, landmark_loss, pose_reg_loss, end_points
            else:
                cls_prob = tf.nn.softmax(logits=cls_prob)
                return cls_prob, bbox_pred, landmark_pred, pose_reg_pred
