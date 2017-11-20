#!/usr/bin/env bash

# This script make tfrecords by multi-threads:
#
# Usage:
# ./scripts/make_tfrecords.sh
IMAGE_SIZE=12
ROOT_PATH='/home/dafu/data/jdap_data/'
LABEL_FILE="/home/dafu/data/jdap_data/${IMAGE_SIZE}/train_${IMAGE_SIZE}.txt"
python ./prepare_data/multithread_create_tfrecords.py \
    --image_root_path=${ROOT_PATH} \
    --dataset_file=${LABEL_FILE} \
    --dataset_name='pnet_aug' \
    --output_dir='./tfrecords' \
    --num_shards=4 \
    --num_threads=4 \
    --image_size=12 \
    --is_shuffle=True