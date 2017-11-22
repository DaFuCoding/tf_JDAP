import tensorflow as tf
import numpy as np
import argparse
import sys
import os
from nets.JDAP_Net import JDAP_12Net as P_Net
from nets.JDAP_Net import JDAP_24Net as R_Net
from nets.JDAP_Net import JDAP_24Net_ERC as R_Net_ERC
from nets.JDAP_Net import JDAP_48Net as O_Net
from nets.JDAP_Net import JDAP_48Net_Lanmark_Pose as O_AUX_Net
from FcnDetector import FcnDetector
from JDAPDetect import JDAPDetector
from Detector import Detector
import cv2
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

FLAGS = tf.flags.FLAGS
FLAGS.ERC_thresh = 0.1


def test_net(name_list, dataset_path, prefix, epoch, batch_size, test_mode="rnet", thresh=[0.6, 0.6, 0.7],
             min_face_size=24, stride=2, shuffle=False, vis=False):

    detectors = [None, None, None]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        RNet = Detector(R_Net_ERC, 24, batch_size[1], model_path[1], aux_idx=4)
        #RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = JDAPDetector(detectors=detectors, is_ERC=True, min_face_size=min_face_size, stride=stride, threshold=thresh)

    fin = open(name_list, 'r')
    #fout = open('/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/pnet/p_%d.txt' % FLAGS.ERC_thresh, 'w')
    fout = open('/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/pnet/pnet_wide.txt', 'w')
    lines = fin.readlines()
    scale = -0.3
    count = 0
    for line in lines:
        count += 1
        # if count % 19 != 0:
        #     continue
        if count % 300 == 0:
            print("Detect %d images." % count)
        related_name = line.strip().split()[0]
        if '.jpg' not in related_name:
            related_name += '.jpg'
        image_name = os.path.join(dataset_path, related_name)
        image = cv2.imread(image_name)
        all_boxes = mtcnn_detector.detect(image)
        box_num = all_boxes.shape[0]
        write_str = line + str(box_num) + '\n'
        for i in range(box_num):
            bbox = all_boxes[i, :4]
            score = all_boxes[i, 4]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            center = ((x1 + x2) / 2, (y1 + y2) /2)
            w = x2 - x1
            h = y2 - y1
            new_x1 = int(x1 - w * scale)
            new_x2 = int(x2 + w * scale)
            new_y1 = int(y1 - h * scale)
            new_y2 = int(y2 + h * scale)
            write_str += ' '.join(str(x) for x in [x1, y1, x2 - x1 + 1, y2 - y1 + 1]) + ' %.4f' % score + '\n'
            if vis:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 2)
                #cv2.rectangle(image, (new_x1, new_y1), (new_x2, new_y2), (200, 200, 0), 2)
        fout.write(write_str)
        if vis:
            cv2.imshow("a", image)
            cv2.waitKey(0)


def test_aux_net(name_list, dataset_path, prefix, epoch, batch_size, test_mode="rnet", thresh=[0.6, 0.6, 0.7],
             min_face_size=24, stride=2, shuffle=False, vis=False):

    detectors = [None, None, None]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        ONet = Detector(O_AUX_Net, 48, batch_size[2], model_path[2], aux_idx=3)
        detectors[2] = ONet

    mtcnn_detector = JDAPDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, threshold=thresh)

    fin = open(name_list, 'r')
    fout = open('/home/dafu/workspace/FaceDetect/tf_JDAP/evaluation/onet/onet_wider_landmark_pose_test.txt', 'w')
    lines = fin.readlines()
    test_image_id = 0
    for line in lines:
        test_image_id += 1
        related_name = line.strip().split()[0]
        if '.jpg' not in related_name:
            related_name += '.jpg'
        print(test_image_id, related_name)
        image_name = os.path.join(dataset_path, related_name)
        image = cv2.imread(image_name)
        src_boxes, cal_boxes, land_reg, pose_reg = mtcnn_detector.detect(image, aux_idx=3)
        box_num = cal_boxes.shape[0]
        write_str = line + str(box_num) + '\n'
        proposal_side = src_boxes[:, 2] - src_boxes[:, 0]
        pose_reg = pose_reg * 180/3.14
        for i in range(box_num):
            bbox = cal_boxes[i, :4]
            score = cal_boxes[i, 4]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            write_str += ' '.join(str(x) for x in [x1, y1, x2 - x1 + 1, y2 - y1 + 1]) + ' %.4f' % score + '\n'
            if vis:
                pose_info = "%.2f %.2f %.2f" % (pose_reg[i][0], pose_reg[i][1], pose_reg[i][2])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 2)
                cv2.putText(image, pose_info, (x1, y2), 1, 1, (200, 200, 0), 2)
                for land_id in range(5):
                    #point_x = int(land_reg[i][land_id * 2] * proposal_side[i] + src_boxes[i][0])
                    #point_y = int(land_reg[i][land_id * 2 + 1] * proposal_side[i] + src_boxes[i][1])
                    point_x = int(land_reg[i][land_id * 2] * w + x1)
                    point_y = int(land_reg[i][land_id * 2 + 1] * h + y1)
                    cv2.circle(image, (point_x, point_y), 2, (200, 0, 0), 2)
        fout.write(write_str)
        if vis:
            cv2.imshow("a", image)
            cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name_list', dest='name_list', help='output data folder',
                        default='/home/dafu/data/FDDB/RYFdefined/FDDBNameList.txt',
                        #default='/home/dafu/data/WIDER_FACE/wider_face_split/wider_name_list.txt',
                        #default='/home/dafu/data/AFLW/data/aflw_rect_pose.txt',
                        type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='/home/dafu/data/FDDB',
                        #default='/home/dafu/data/WIDER_FACE/WIDER_train/images',
                        #default='/home/dafu/data/AFLW/data',
                        type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='pnet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=[
                                # PNet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_FL/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_pos_neg/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_retrain/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_more_c/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wider_OHEM_0.7/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_aug/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_aug_MC/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_aug_LB/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wider_OHEM_0.7_LB/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wider_OHEM_0.7_wo_pooling/pnet',
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_order_SB/pnet',
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_OHEM_0.7_shuffle_SB/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_wider_FL/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_test/pnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/pnet/pnet_test_FL_relu/pnet',
                                # RNet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_wider_OHEM_0.7/rnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_ERC_keep_all_pos/rnet',
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_ERC/rnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_ERC_pos_neg/rnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_ERC_LB/rnet',
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/rnet/rnet_wider_OHEM_0.7_head/rnet',
                                # ONet
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_SF/onet'
                                #'/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_OHEM_0.7_FL_gamma2/onet'
                                '/home/dafu/workspace/FaceDetect/tf_JDAP/models/onet/onet_wider_landmark_pose_OHEM_0.7/onet'
                                ], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[6, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.6, 0.1, 0.1], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', default=True, help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)
    test_net(args.name_list, args.dataset_path, args.prefix,
             args.epoch, args.batch_size, args.test_mode,
             args.thresh, args.min_face, args.stride, args.shuffle, args.vis)

    # test_aux_net(args.name_list, args.dataset_path, args.prefix,
    #              args.epoch, args.batch_size, args.test_mode,
    #              args.thresh, args.min_face, args.stride, args.shuffle, args.vis)