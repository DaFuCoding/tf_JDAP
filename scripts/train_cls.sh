#!/usr/bin/env bash

# This script train classification task:
#
# Usage:
# ./scripts/train_cls.sh

IMAGE_SIZE=48
STAGE_NAME='onet'
RECORD_ROOT='/home/dafu/workspace/FaceDetect/tf_JDAP/tfrecords'
#--image_sum=1031327 \
# pnet train: 1479686 val: 271973
# rnet train: 660181  val: 120364
# rnet_add_gt train: 1015558  val: 120364
# onet_add_gt train: 1186982
python ./train/train_cls.py \
    --gpu_id='0' \
    --image_sum=1186982 \
    --val_image_sum=120364 \
    --logdir="./logdir/${STAGE_NAME}/${STAGE_NAME}_wider_OHEM_0.7_mnet_relu6" \
    --loss_type='SF' \
    --is_ohem=True \
    --is_ERC=False \
    --model_prefix="./models/${STAGE_NAME}/${STAGE_NAME}_wider_OHEM_0.7_mnet_relu6/${STAGE_NAME}" \
    --tfrecords_root="${RECORD_ROOT}/${STAGE_NAME}/${STAGE_NAME}" \
    --tfrecords_num=4 \
    --image_size=${IMAGE_SIZE} \
    --frequent=500 \
    --batch_size=128 \
    --end_epoch=16 \
    --lr=0.01 \
    --lr_decay_factor=0.1 \
    --optimizer='momentum' \
    --momentum=0.9
