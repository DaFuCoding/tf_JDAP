import argparse
import os.path as osp
from prepare_data.data_base import *

aflw2000_root = ''


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation performance',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', dest='dataset_name', help='dataset_name',
                        #default='fddb',
                        #default='celebA',
                        #default='wider',
                        default='300wp',
                        #default='ls3dw',
                        type=str)

    parser.add_argument('--name_list', dest='name_list', help='output data folder',
                        #default='/home/dafu/data/FDDB/RYFdefined/FDDBNameList.txt',
                        #default='/home/dafu/data/FDDB/RYFdefined/test.txt',
                        #default='/home/dafu/data/CelebA/list_eval_partition.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_train_annotation.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_val_annotation.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_test_name.txt',
                        #default='/home/dafu/data/AFW/Annotation/AFWNameList.txt',
                        #default='/home/dafu/data/AFLW/data/aflw_rect_pose.txt',
                        #default='/home/dafu/data/300W-LP/300WP_pose.txt',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/menpo_test.txt',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/aflw2000_3d_gt.txt',
                        default='/home/dafu/data/AFLW2000_21pts_vis/Code/aflw2000_3d_gt.txt',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/test.txt',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/300VW-3D_file_name.txt',
                        type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        #default='/home/dafu/data/FDDB',
                        #default='/home/dafu/data/CelebA/img_celeba',
                        #default='/home/dafu/data/WIDER_FACE/WIDER_train/images',
                        #default='/home/dafu/data/WIDER_FACE/WIDER_val/images',
                        #default='/home/dafu/data/WIDER_FACE/WIDER_test/images',
                        #default='/home/dafu/data/AFW',
                        #default='/home/dafu/data/AFLW/data',
                        #default='/home/dafu/data/300W-LP',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/Menpo-3D',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/AFLW2000-3D-Reannotated',
                        default='/home/dafu/data/AFLW2000_21pts_vis',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/300W-Testset-3D',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/300VW-3D',
                        type=str)
    parser.add_argument('--eval_file', dest='eval_file', help='eval file name',
                        # landmark 68 and pose
    default='/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/aflw2000/onet_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_1w_mean_shape_16_0.1_0.01_0.01.txt',
    #default='/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/aflw2000/onet_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_0.1w_16_0.1_0.01_0.01.txt',
                        # landmark 7 and pose
    #default='/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/aflw2000/onet_OHEM_0.7_wop_pnet_300WLP_pose_landmark7_1w_16_0.1_0.01_0.01.txt',

                        # single task
    #default='/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/aflw2000/onet_OHEM_0.7_wop_pnet_300WLP_landmark68_0.1w_16_0.1_0.01_0.01.txt',
    #default='/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/aflw2000/onet_OHEM_0.7_wop_pnet_300WLP_pose_16_0.1_0.01_0.01.txt',

                        type=str)

    args = parser.parse_args()
    return args
def draw_rectangle(image, rect, color=(0, 0, 200)):
    rect = [int(x) for x in rect]
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

from demo.detectAPI import test_single_net
import tensorflow as tf
FLAGS = tf.flags.FLAGS
FLAGS.landmark_num = 68
from tools.utils import *
if __name__ == '__main__':
    args = parse_args()

    if args.dataset_name == 'ls3dw':
        brew_fun = LS3DW
    elif args.dataset_name == '300wp':
        brew_fun = L300WP
    else:
        print(NotImplementedError)

    Evaluator = brew_fun(args.dataset_path, args.name_list)
    detector = test_single_net('../models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_0.1w/onet',
                               16, 48, 'landmark_pose')
    results = Evaluator.eval_accuracy(args.eval_file, eval_dim=68, eval_mode='landmark_pose', detector=detector,
                                      vis=False)





