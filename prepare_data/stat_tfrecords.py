from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _gen_features(vector_name, vector_dim):
    """
    img_raw and cls_label is default
    Args:
        vector_name: other feature name
        vector_dim: other feature dim

    Returns:
        features dic
    """
    features = {'img_raw': tf.FixedLenFeature([], tf.string),
                'cls_label': tf.FixedLenFeature([], tf.int64)}
    if type(vector_name) is list:
        for name, dim in zip(vector_name, vector_dim):
            features[name] = tf.FixedLenFeature([dim], tf.float32)
    else:
        features[vector_name] = tf.FixedLenFeature([vector_dim], tf.float32)
    return features


def _features_processing(features, vector_name, imgSide, channels, data_format='NHWC'):
    """
    processing img
    Args:
        features: parse single example
        vector_name: other vector name

    Returns:
        processed dic
    """

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    if data_format is not 'NHWC':
        raise Exception("Not support this data format.")
    img = tf.reshape(img, [imgSide, imgSide, channels])
    img = (tf.cast(img, tf.float32) - 127.5) * 0.0078125
    cls_label = tf.cast(features['cls_label'], tf.int64)
    data_dic = {'image': img, 'cls_label': cls_label}
    if type(vector_name) is list:
        for name in vector_name:
            data_dic[name] = tf.cast(features[name], tf.float32)
    else:
        data_dic[vector_name] = tf.cast(features[vector_name], tf.float32)
    return data_dic


def ReadTFRecord(tfrecords, imgSide, channels, data_format='NHWC',
                 vector_name='reg_label', vector_dim=4):
    """Read one or more tfrecords
    Args:
      tfrecords: the list of many tfrecords
      imgSide: square image side
      channels: image channels
      data_format: NHWC(default) like TF, NCHW like Caffe
      vector_name: parsing vector name
        reg_label(4), head_pose_label(3) and landmark_label(Nx2)
      vector_dim: vector dimension
    :return img_raw, cls_label, reg_label
    """
    record_queue = tf.train.string_input_producer(tfrecords, shuffle=True)
    reader = tf.TFRecordReader()

    _, serialized_ex = reader.read(record_queue)

    # features = tf.parse_single_example(serialized_ex,
    #         features={
    #             'img_raw': tf.FixedLenFeature([], tf.string),
    #             'cls_label': tf.FixedLenFeature([], tf.int64),
    #             vector_name: tf.FixedLenFeature([vector_dim], tf.float32)
    #         })
    features = tf.parse_single_example(serialized_ex, features=_gen_features(vector_name, vector_dim))
    return _features_processing(features, vector_name, imgSide, channels, data_format)
    # img = tf.decode_raw(features['img_raw'], tf.uint8)
    # if data_format is not 'NHWC':
    #     raise Exception("Not support this data format.")
    # img = tf.reshape(img, [imgSide, imgSide, channels])
    # img = (tf.cast(img, tf.float32) - 127.5) * 0.0078125
    #
    # cls_label = tf.cast(features['cls_label'], tf.int64)
    # vector_label = tf.cast(features[vector_name], tf.float32)
    # return {'image': img, 'cls_label': cls_label, vector_name: vector_label}