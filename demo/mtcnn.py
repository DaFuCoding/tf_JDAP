import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import cv2

from detectAPI import DetectAPI
from prepare_data.data_base import FDDB
from prepare_data.data_base import L300WP
from prepare_data.data_base import LS3DW
from prepare_data.data_base import WIDER

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

FLAGS = tf.flags.FLAGS
FLAGS.ERC_thresh = 0.1
FLAGS.landmark_num = 68


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', dest='dataset_name', help='dataset_name',
                        #default='fddb',
                        #default='celebA',
                        #default='wider',
                        #default='300wp',
                        default='ls3dw',
                        type=str)

    parser.add_argument('--name_list', dest='name_list', help='output data folder',
                        #default='/home/dafu/data/FDDB/RYFdefined/FDDBNameList.txt',
                        #default='/home/dafu/data/FDDB/RYFdefined/test.txt',
                        #default='/home/dafu/data/CelebA/list_eval_partition.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_train_annotation.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_val_annotation.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_test_name.txt',
                        #default="/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/wider-test-image/wider-test-image.txt",
                        #default='/home/dafu/data/AFW/Annotation/AFWNameList.txt',
                        #default='/home/dafu/data/AFLW/data/aflw_rect_pose.txt',
                        #default='/home/dafu/data/300W-LP/300WP_pose.txt',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/menpo_test.txt',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/aflw2000_3d_gt.txt',
                        #default='/home/dafu/data/AFLW2000_21pts_vis/Code/aflw2000_3d_gt.txt',
                        #default='/media/dafu/DAFU/paper/AFLW_2000_Pose/yaw_image_name.txt',
                        #default='/home/dafu/data/AFLW2000_21pts_vis/Code/AFLW2000_test.txt',
                        default='/home/dafu/data/LS3D-W/LS3D-W/test.txt',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/300VW-3D_file_name.txt',
                        #default='/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/MultiScale/exp.txt',
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
                        #default='/home/dafu/data/AFLW2000_21pts_vis',
                        default='/home/dafu/data/LS3D-W/LS3D-W/300W-Testset-3D',
                        #default='/home/dafu/data/LS3D-W/LS3D-W/300VW-3D',
                        #default='/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/MultiScale',
                        type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='onet_landmark_pose', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=[
                                # PNet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wop_relu6/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wop_relu6_addval/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_wo_pooling/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wider_OHEM_0.7/pnet',
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_wo_pooling/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_1.0_wo_pooling/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_wo_pooling_wo_part/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_wop_retrain/pnet',

                                # MNet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/mnet/mnet_wider_OHEM_0.7_wop_pnet_add_gt/mnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/mnet/mnet_wider_OHEM_0.7_wop_pnet_add_gt_all_relu6/mnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/mnet/mnet_OHEM_0.7_wop_pnet_all_mp2s_relu6/mnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/mnet/mnet_OHEM_0.7_wop_pnet_all_mp2s_prelu/mnet',
                                # RNet
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_wider_OHEM_0.7_wop_pnet/rnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_OHEM_0.7_wop_pnet_wo_part/rnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_wider_OHEM_0.7_wop_pnet_add_gt/rnet',
                                # ONet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_test/onet',
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_1w_mean_shape/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_1w_dynamic_shape/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_1w_yaw_bin35_dynamic_shape/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_1w_range_dynamic_shape/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_0.1w/onet',

                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark68_0.1w_landmark_OHEM_0.7/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_landmark68_0.1w/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_landmark68_1w_mean_shape/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark51_0.1w/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_landmark7/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_landmark7/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_last_conv_branch64/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_wop_pnet_300WLP_pose_yaw/onet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_OHEM_0.7_wop_pnet/onet'
                                # ANet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/anet/anet_wider_OHEM_0.7_mnet_300WLP/anet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/anet/anet_wider_OHEM_0.7_mnet_only_cls/anet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/anet/anet_wider_OHEM_0.7_mnet_300WLP_dp0.75/anet',

                        ], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[13, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.4, 0.2, 0.1], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=10, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)
    #output_file = '/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/aflw2000/onet_OHEM_0.7_wop_pnet_300WLP_pose_yaw_16_0.1_0.01_0.01.txt'
    output_file = '/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/onet/onet_OHEM_0.7_wop_pnet_300WLP_landmark68_1w_mean_shape_16_0.4_0.1_0.01.txt'
    output = False
    detector = DetectAPI(args.prefix, args.epoch, args.test_mode, args.batch_size,
                         is_ERC=False, thresh=args.thresh, min_face_size=args.min_face)
    mode = 'val'
    is_wider = False
    # Select data set
    if args.dataset_name == 'fddb':
        brew_fun = FDDB
    elif args.dataset_name == 'wider':
        brew_fun = WIDER
        output_dir = '/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/wider_val_pred/pose_landmark68_0.1w'
        is_wider = True
    elif args.dataset_name == '300wp':
        brew_fun = L300WP
    elif args.dataset_name == 'ls3dw':
        brew_fun = LS3DW

    Evaluator = brew_fun(args.dataset_path, args.name_list)
    is_cap = False

    import time

    count = 1
    if is_cap:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            while (True):
                # get a frame
                ret, frame = cap.read()
                t = time.clock()
                results = detector.detect(frame)
                print("time: %d ms" % int((time.clock() - t) * 1000))
                # Return detect result adapt to evaluation
                if type(results) == np.ndarray and len(results):
                    detector.show_result(frame, results, is_cap=True)
                elif type(results) == tuple and results[0] is not None:
                    detector.show_result(frame, results, is_cap=True)

                # show a frame
                cv2.imshow("capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
    elif is_wider:
        for label_info in Evaluator.label_infos:
            count += 1
            # if count % 10:
            #     continue
            if count % 100 == 0:
                print(count)
            image_name = Evaluator.get_image_name(label_info)
            # print(image_name)
            image = cv2.imread(image_name)
            # All output result
            results = detector.detect(image)
            Evaluator.do_eval(label_info, results, output_dir)
    else:
        save_dir = '/home/dafu/Pictures/300W-Testset-3D'
        # Demo test
        if output:
            fout = open(output_file, 'w')
        for label_info in Evaluator.label_infos:
            count += 1
            # if count % 15:
            #     continue
            if count % 100 == 0:
                print(count)
            image_name = Evaluator.get_image_name(label_info)
            print(image_name)
            image = cv2.imread(image_name)
            # All output result
            results = detector.detect(image)
            if output:
                #fout.write(Evaluator.do_eval(label_info, results, 'pose'))
                fout.write(Evaluator.do_eval(label_info, results))
            elif (type(results) == np.ndarray and len(results)) or (type(results) == tuple and results[0] is not None):
                # Return detect result adapt to evaluation
                if save_dir != '':
                    detector.show_result(image, results, os.path.join(save_dir, os.path.basename(image_name)))
                else:
                    detector.show_result(image, results)
        if output:
            fout.close()
