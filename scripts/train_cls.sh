#!/usr/bin/env bash

# This script train classification task:
#
# Usage:
# ./scripts/train_cls.sh

IMAGE_SIZE=12
STAGE_NAME='pnet'
RECORD_ROOT='/home/dafu/workspace/FaceDetect/tf_JDAP/tfrecords'
#--image_sum=1031327 \
# pnet_aug 1618753
python ./train/train_cls.py \
    --gpu_id='0' \
    --image_sum=1618753 \
    --logdir="./logdir/${STAGE_NAME}/${STAGE_NAME}_aug_MC" \
    --loss_type='SF' \
    --is_ohem=True \
    --is_ERC=False \
    --model_prefix="./models/${STAGE_NAME}/${STAGE_NAME}_aug_MC/${STAGE_NAME}" \
    --tfrecords_root="${RECORD_ROOT}/${STAGE_NAME}_aug" \
    --tfrecords_num=4 \
    --image_size=${IMAGE_SIZE} \
    --frequent=500 \
    --batch_size=128 \
    --end_epoch=16 \
    --lr=0.01 \
    --lr_decay_factor=0.1 \
    --optimizer='momentum' \
    --momentum=0.9