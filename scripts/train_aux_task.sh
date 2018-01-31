#!/usr/bin/env bash

# This script train auxiliary task:
#
# Usage:
# ./scripts/train_aux_task.sh

IMAGE_SIZE=48
STAGE_NAME='onet'
RECORD_ROOT='/home/dafu/workspace/FaceDetect/tf_JDAP/tfrecords'
RECORD_NAME='onet'
# onet_wop_pnet train: 986545
# 300WLP tran: 244454
python ./train/train_auxiliary.py \
    --gpu_id='0' \
    --image_sum=986545 \
    --val_image_sum=120364 \
    --logdir="./logdirs/${STAGE_NAME}/${STAGE_NAME}_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_1w_yaw_bin35__dynamic_shape" \
    --loss_type='SF' \
    --is_ohem=True \
    --is_ERC=False \
    --model_prefix="./models/${STAGE_NAME}/${STAGE_NAME}_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_1w_yaw_bin35_dynamic_shape/${STAGE_NAME}" \
    --tfrecords_root="${RECORD_ROOT}/${RECORD_NAME}/${RECORD_NAME}" \
    --tfrecords_num=4 \
    --image_size=${IMAGE_SIZE} \
    --frequent=200 \
    --batch_size=128 \
    --end_epoch=16 \
    --lr=0.01 \
    --lr_decay_factor=0.1 \
    --optimizer='momentum' \
    --momentum=0.9 \
    --is_feature_visual=False


