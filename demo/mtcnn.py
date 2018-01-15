import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import cv2

from detectAPI import DetectAPI
from prepare_data.data_base import FDDB

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
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_redundant/pnet',
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_wo_pooling/pnet',
                                # RNet
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_wider_OHEM_0.7/rnet',
                                # ONet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_SF/onet'
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_landmark_pose_OHEM_0.7/onet',
                        ], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[13, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.4, 0.1, 0.1], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)
    output_file = '/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/pnet/pnet_OHEM_0.7_wo_pooling.txt'

    detector = DetectAPI(args.prefix, args.epoch, args.test_mode, args.batch_size,
                         is_ERC=False, thresh=args.thresh, min_face_size=args.min_face)
    # Select data set
    if args.dataset_name == 'fddb':
        brew_fun = FDDB
    elif args.dataset_name == 'wider':
        pass
    elif args.dataset_name == '300wp':
        pass

    Evaluator = brew_fun(args.dataset_path, args.name_list)
    # Demo test
    for label_info in Evaluator.label_infos:
        image_name = Evaluator.label_parser(label_info)
        image = cv2.imread(image_name)
        # All output result
        results = detector.detect(image)
        # Return detect result adapt to evaluation
        #print(Evaluator.do_eval(label_info, results))
        detector.show_result(image, results)
