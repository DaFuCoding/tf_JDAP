#!/usr/bin/env python
"""
Create the tfrecord files for a dataset in multi-threading.
Author: Ren Yanfu
"""
from __future__ import absolute_import, division, print_function

import sys
from datetime import datetime
import os
import random
import threading
import cv2
import numpy as np
import tensorflow as tf

# Auxiliary function to convert any value to list type
def _toList(value):
    return value if type(value) == list else [value]

def _int64_feature(value):
    value = _toList(value)
    value = [int(x) for x in value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    value = _toList(value)
    value = [float(x) for x in value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=_toList(value)))


flags = tf.app.flags

flags.DEFINE_string("image_root_path", "/home/dafu/data/jdap_data/", "image root path")
flags.DEFINE_string(
    "dataset_file", "/home/dafu/data/jdap_data/12/test_12.txt",
    "label file about making tfrecords.")
flags.DEFINE_string("dataset_name", "pnet_test", "Ouput tfrecords' name")
flags.DEFINE_string("output_dir", "./tfrecords", "TFRecords directry of output.")
flags.DEFINE_integer("num_shards", 4, "Number of shards to make.")
flags.DEFINE_integer("num_threads", 4, "Number of threads to make.")
flags.DEFINE_integer("image_size", 12, "Image size.")
flags.DEFINE_boolean("is_shuffle", True, "Shuffle the records before saving them.")

FLAGS = flags.FLAGS


class TFRecord(object):
    """A concrete class for fast make TFRecord by using multi-thread

    """
    def __init__(self, image_root_path, label_file, dataset_name, output_dir,
                 num_shards, num_threads, image_size, is_shuffle=True):
        """
        TODO: Attention: must num_threads <= num_shards!!!
        Args:
            image_root_path: image root path
            label_file:
            dataset_name: tfrecord name
            output_dir: directory of saving tfrecord
            num_shards: quantity of shards output
            num_threads: using number of threads
            image_size: image size(row equal column)
            is_shuffle: whether shuffle
        """
        self._image_root_path = image_root_path
        self._label_file = label_file
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._num_shards = num_shards
        self._num_threads = num_threads
        self._image_size = image_size
        self._is_shuffle = is_shuffle

    def _set_single_example(self, image_example, image_path):
        img = cv2.imread(image_path)
        h, w, c = img.shape
        if not (h == w and w == self._image_size):
            img = cv2.resize(img, (self._image_size, self._image_size), interpolation=cv2.INTER_LINEAR)
        img_raw = img.tostring()
        example = tf.train.Example(
            # Create a Features
            features=tf.train.Features(
                # custom define key-value
                feature={
                    "img_raw": _bytes_feature(img_raw),
                    "cls_label": _int64_feature(image_example[1]),
                    "reg_label": _float_feature(image_example[2:6]),
                    # "landmark_label": _float_feature(image_example[2:12]),
                    # "pose_label": _float_feature(image_example[2:5]),
                }
            ))
        return example

    def _process_image_files_batch(self, thread_index, ranges):
        """Processes and saves list of images as TFRecord in 1 thread.
        Args:
          thread_index: integer, unique batch to run index is within [0, len(ranges)).
          ranges: list of pairs of integers specifying ranges of each batches to
            analyze in parallel.
        """
        # Each thread produces N shards where N = int(num_shards / num_threads).
        # For instance, if num_shards = 128, and the num_threads = 2, then the first
        # thread would produce shards [0, 64).
        num_threads = len(ranges)
        assert not self._num_shards % num_threads
        num_shards_per_batch = int(self._num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                                   num_shards_per_batch + 1).astype(int)
        num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

        counter = 0
        error_counter = 0
        error_image_name = []
        for s in xrange(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s
            output_filename = '%s-%.5d-of-%.5d' % (self._dataset_name, shard, self._num_shards)
            output_file = os.path.join(self._output_dir, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)

            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in files_in_shard:
                image_example = self._dataset[i].strip().split()
                image_path = os.path.join(self._image_root_path, image_example[0])
                try:
                    example = self._set_single_example(image_example, image_path)
                    writer.write(example.SerializeToString())
                    shard_counter += 1
                    counter += 1
                except Exception as e:
                    print("image error")
                    error_counter += 1
                    error_image_name.append(image_path)

                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch, with %d errors.' %
                          (datetime.now(), thread_index, counter, num_files_in_thread, error_counter))
                    sys.stdout.flush()

            print('%s [thread %d]: Wrote %d images to %s, with %d errors.' %
                  (datetime.now(), thread_index, shard_counter, output_file, error_counter))
            sys.stdout.flush()
        print('%s [thread %d]: Wrote %d images to %d shards, with %d errors.' %
              (datetime.now(), thread_index, counter, num_files_in_thread, error_counter))
        print("error image path : ", error_image_name)
        sys.stdout.flush()

    def create(self):
        self._dataset = open(self._label_file, 'r').readlines()
        # Images in the tfrecords set must be shuffled properly
        if self._is_shuffle:
            random.shuffle(self._dataset)
        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        spacing = np.linspace(0, len(self._dataset), self._num_threads + 1).astype(np.int)
        ranges = []
        for i in xrange(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i+1]])

        # Launch a thread for each batch.
        print('Launching %d threads for spacings: %s' % (self._num_threads, ranges))
        sys.stdout.flush()

        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()

        threads = []
        for thread_index in xrange(len(ranges)):
            args = (thread_index, ranges)
            t = threading.Thread(target=self._process_image_files_batch, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(self._dataset)))


def main(_):

    record = TFRecord(FLAGS.image_root_path, FLAGS.dataset_file, FLAGS.dataset_name,
                      FLAGS.output_dir, FLAGS.num_shards, FLAGS.num_threads, FLAGS.image_size, FLAGS.is_shuffle)
    record.create()


if __name__ == '__main__':
    tf.app.run()