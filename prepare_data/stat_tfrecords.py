"""
These utility functions are meant for computing basic statistics in a set of tfrecord
files. They can be used to sanity check the training and testing files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf


def ReadTFRecord(tfrecords, imgSide, channels, vector_name, vector_num):
    """Read one or more tfrecords
    Args:
      tfrecords: the list of many tfrecords
      imgSide: square image side
      channels: image channels
      vector_name: parsing vector name
      reg_label(4), pose_label(3) and landmark_label(10)
      vector_num: vector dimension
    :return img_raw, cls_label, reg_label
    """
    record_queue = tf.train.string_input_producer(tfrecords, shuffle=True)
    reader = tf.TFRecordReader()

    _, serialized_ex = reader.read(record_queue)

    features = tf.parse_single_example(serialized_ex,
            features={
                'img_raw': tf.FixedLenFeature([], tf.string),
                'cls_label': tf.FixedLenFeature([], tf.int64),
                vector_name: tf.FixedLenFeature([vector_num], tf.float32)
            })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [imgSide, imgSide, channels])
    img = (tf.cast(img, tf.float32) - 127.5) * 0.0078125

    cls_label = tf.cast(features['cls_label'], tf.int64)
    vector_label = tf.cast(features[vector_name], tf.float32)
    return {'image': img, 'cls_label': cls_label, vector_name: vector_label}


def _ReadTFRecord(tfrecords, imgSide, channels):
    """Read one or more tfrecords
    Args:
      tfrecords: the list of many tfrecords
      imgSide: square image side
      channels: image channels
    :return img_raw, cls_label, reg_label
    """
    record_queue = tf.train.string_input_producer(tfrecords, shuffle=True)
    reader = tf.TFRecordReader()

    _, serialized_ex = reader.read(record_queue)

    features = tf.parse_single_example(serialized_ex,
            features={
                'img_raw': tf.FixedLenFeature([], tf.string),
                'cls_label': tf.FixedLenFeature([], tf.int64),
                'reg_label': tf.FixedLenFeature([4], tf.float32)
            })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [imgSide, imgSide, channels])
    img = (tf.cast(img, tf.float32) - 127.5) * 0.0078125

    cls_label = tf.cast(features['cls_label'], tf.int64)
    reg_label = tf.cast(features['reg_label'], tf.float32)
    return img, cls_label, reg_label


def ReadPoseTFRecord(tfrecords, imgSide, channels):
    """Read one or more face pose tfrecords
    Args:
      tfrecords: the list of many tfrecords
      imgSide: square image side
      channels: image channels
    :return img_raw, cls_label, reg_label
    """
    record_queue = tf.train.string_input_producer(tfrecords, shuffle=True)
    reader = tf.TFRecordReader()

    _, serialized_ex = reader.read(record_queue)

    features = tf.parse_single_example(serialized_ex,
            features={
                'img_raw': tf.FixedLenFeature([], tf.string),
                'cls_label': tf.FixedLenFeature([], tf.int64),
                'pose_label': tf.FixedLenFeature([3], tf.float32)
            })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [imgSide, imgSide, channels])
    img = (tf.cast(img, tf.float32) - 127.5) * 0.0078125

    cls_label = tf.cast(features['cls_label'], tf.int64)
    pose_label = tf.cast(features['pose_label'], tf.float32)
    return img, cls_label, pose_label


def ReadLandmarkTFRecord(tfrecords, imgSide, channels):
    """Read one or more face pose tfrecords
    Args:
      tfrecords: the list of many tfrecords
      imgSide: square image side
      channels: image channels
    :return img_raw, cls_label, reg_label
    """
    record_queue = tf.train.string_input_producer(tfrecords, shuffle=True)
    reader = tf.TFRecordReader()

    _, serialized_ex = reader.read(record_queue)

    features = tf.parse_single_example(serialized_ex,
            features={
                'img_raw': tf.FixedLenFeature([], tf.string),
                'cls_label': tf.FixedLenFeature([], tf.int64),
                'landmark_label': tf.FixedLenFeature([10], tf.float32)
            })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [imgSide, imgSide, channels])
    img = (tf.cast(img, tf.float32) - 127.5) * 0.0078125

    cls_label = tf.cast(features['cls_label'], tf.int64)
    landmark_label = tf.cast(features['landmark_label'], tf.float32)
    return img, cls_label, landmark_label


def class_stats(tfrecords, image_size, channels, vector_name, vector_num):
    cls_data_label = ReadTFRecord(tfrecords, image_size, channels, vector_name, vector_num)
    images = cls_data_label['image']
    cls_labels = cls_data_label['cls_label']
    reg_labels = cls_data_label['reg_label']
    sess = tf.Session()
    img = sess.run(images)
    print(img)


def parse_args():

    parser = argparse.ArgumentParser(description='Basic statistics on tfrecord files')

    parser.add_argument('--stat', dest='stat_type',
                        choices=['class_stats', 'verify_bboxes'],
                        required=True)

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)


    parsed_args = parser.parse_args()

    return parsed_args


def main():
    dataset_name = '../tfrecords/pnet'
    tfrecords = []
    for i in range(4):
        tfrecords.append("%s-%.5d-of-00005" % (dataset_name, i))
    class_stats(tfrecords, 12, 3, 'reg_label', 4)


if __name__ == '__main__':
    main()