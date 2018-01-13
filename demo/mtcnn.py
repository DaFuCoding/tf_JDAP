import argparse
import sys
import os
import math
import glob

import tensorflow as tf
import numpy as np
import cv2

from detectAPI import FDDB

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

FLAGS = tf.flags.FLAGS
FLAGS.ERC_thresh = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', dest='dataset_name', help='dataset_name',
                        default='fddb',
                        #default='celebA',
                        #default='wider',
                        #default='300wp',
                        type=str)

    parser.add_argument('--name_list', dest='name_list', help='output data folder',
                        default='/home/dafu/data/FDDB/RYFdefined/FDDBNameList.txt',
                        #default='/home/dafu/data/CelebA/list_eval_partition.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_name_list.txt',
                        #default='/home/dafu/data/AFLW/data/aflw_rect_pose.txt',
                        #default='/home/dafu/data/300W-LP/300WP_pose.txt',
                        type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='/home/dafu/data/FDDB',
                        #default='/home/dafu/data/CelebA/img_celeba',
                        #default='/home/dafu/data/WIDER_FACE/WIDER_train/images',
                        #default='/home/dafu/data/AFLW/data',
                        #default='/home/dafu/data/300W-LP',
                        type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='pnet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=[
                                # PNet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wider_OHEM_0.7/pnet',
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_redundant/pnet',
                                # RNet
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_wider_OHEM_0.7/rnet',
                                # ONet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_SF/onet'
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_landmark_pose_OHEM_0.7/onet',
                        ], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.4, 0.1, 0.1], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    parser.add_argument('--vis', dest='vis', default=True, help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)
    output_file = '/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/pnet/redundant_0.4_16.txt'

    if args.dataset_name == 'fddb':
        brew_fun = FDDB
    elif args.dataset_name == 'wider':
        pass
    elif args.dataset_name == '300wp':
        pass

    Evaluator = brew_fun(args.dataset_path, args.vis,
                         args.name_list, args.prefix, args.epoch, args.test_mode, args.batch_size,
                         is_ERC=False, thresh=args.thresh, min_face_size=args.min_face)

    Evaluator.do_eval(output_file)
    # for label_info in Evaluator.label_infos:
    #     image_name = Evaluator.label_parser(label_info)
    #     image = cv2.imread(image_name)
    #     # All output result
    #     results = Evaluator.detect(image)
    #     if Evaluator.vis:
    #         Evaluator.show_result(image, results)

    # stage = 48
    # img_list = glob.glob('/home/dafu/Pictures/test/%d_*.jpg' % stage)
    # test_single_net(img_list, '/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_landmark_pose_OHEM_0.7/onet',
    #                 16, stage, True)

    # test_aux_net(args.name_list, args.dataset_path, args.prefix,
    #              args.epoch, args.batch_size, args.test_mode,
    #              args.thresh, args.min_face, args.stride, args.vis)