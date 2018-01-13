import numpy as np
import numpy.random as npr
import argparse
import os
import cPickle
import cv2
import sys
from operator import itemgetter, attrgetter
from configs.config import config
os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(config.gpu_id))
from nets.JDAP_Net import JDAP_12Net as P_Net, JDAP_24Net as R_Net
from demo.fcn_detector import FcnDetector
from demo.jdap_detect import JDAPDetector
from demo.detector import Detector
from configs.config import config
from tools.utils import *

netSize = 48
wider_face_path = '/home/dafu/data/WIDER_FACE/WIDER_train/images'
data_dir = '/home/dafu/data/jdap_data'
anno_file = os.path.join("/home/dafu/data/WIDER_FACE/wider_face_split/wider_train_annotation.txt")
neg_save_dir = os.path.join(data_dir, "%d/negative" % netSize)
pos_save_dir = os.path.join(data_dir, "%d/positive" % netSize)
part_save_dir = os.path.join(data_dir, "%d/part" % netSize)
for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_model(prefix, epoch, batch_size, test_mode="rnet", thresh=[0.6, 0.6, 0.7], min_face_size=24, stride=2):
    detectors = [None, None, None]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet
    mtcnn_detector = JDAPDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, threshold=thresh)
    return mtcnn_detector


def save_head_face_example(netSize, expandScale):
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print "processing %d images in total" % num_of_images

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = annotation[0]

        boxes = map(float, annotation[1:])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = os.path.join(data_dir, "%d" % netSize)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    det_boxes = cPickle.load(open(os.path.join(save_path, 'detections_%d_0.4.pkl' % netSize), 'r'))
    print len(det_boxes), num_of_images
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    f1 = open(os.path.join(save_path, 'head_pos_%d.txt' % netSize), 'w')
    f2 = open(os.path.join(save_path, 'head_neg_%d.txt' % netSize), 'w')
    f3 = open(os.path.join(save_path, 'head_part_%d.txt' % netSize), 'w')
    p_idx = 0
    n_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print "%d images done" % image_done
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(os.path.join(wider_face_path, im_idx))
        image_height, image_width, _ = img.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        dets_w = dets[:, 2] - dets[:, 0]
        dets_h = dets[:, 3] - dets[:, 1]
        head_dets = np.empty([len(dets), 4])
        head_dets[:, 0] = dets[:, 0] - np.round(dets_w * expandScale)
        head_dets[:, 1] = dets[:, 1] - np.round(dets_h * expandScale)
        head_dets[:, 2] = dets[:, 2] + np.round(dets_w * expandScale)
        head_dets[:, 3] = dets[:, 3] + np.round(dets_h * expandScale)

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(head_dets, image_width, image_height)

        for i, (box, head_det) in enumerate(zip(dets, head_dets)):
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IoU(box, gts)
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            resized_head = cv2.resize(tmp, (netSize, netSize), interpolation=cv2.INTER_LINEAR)
            # compute intersection over union(IoU) between current box and all gt boxes

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                if npr.rand(1) > 0.3 or box[4] < 0.8:
                    continue
                save_file = os.path.join(neg_save_dir, "%d.jpg" % n_idx)
                f2.write("%d/head_negative/%s.jpg" % (netSize, n_idx) + ' 0 0 0 0 0 0\n')
                cv2.imwrite(save_file, resized_head)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write("%s/head_positive/%s.jpg" % (netSize, p_idx) + ' 1 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_head)
                    p_idx += 1
                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write("%s/head_part/%s.jpg" % (netSize, d_idx) + ' -1 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_head)
                    d_idx += 1

    f1.close()
    f2.close()
    f3.close()


def save_hard_example(net):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    if net == "24":
        image_size = 24
    if net == "48":
        image_size = 48

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print "processing %d images in total" % num_of_images

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = annotation[0]

        boxes = map(float, annotation[1:])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = os.path.join(data_dir, "%s" % net)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    #f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    det_boxes = cPickle.load(open(os.path.join(save_path, 'detections_%d_0.4_0.1.pkl' % image_size), 'r'))
    print len(det_boxes), num_of_images
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print "%d images done"%image_done
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(os.path.join(wider_face_path, im_idx))
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                # if npr.rand(1) > 0.3 or box[4] < 0.8:
                #     continue
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("%s/negative/%s.jpg" % (net, n_idx) + ' 0 0 0 0 0 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                continue
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write("%s/positive/%s.jpg" % (net, p_idx) + ' 1 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write("%s/part/%s.jpg" % (net, d_idx) + ' -1 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

        # if len(resized_conds) > 60:
        #     neg_keep = npr.choice(len(resized_conds), size=60, replace=False)
        # else:
        #     neg_keep = npr.choice(len(resized_conds), size=len(resized_conds), replace=False)
        # for neg_idx in neg_keep:
        #     if n_idx % folder_part_threshold == 0:
        #         part_n += 1
        #         new_part_path = data_dir + "/%s/negative/part%d" % (net, part_n)
        #         if not os.path.exists(new_part_path):
        #             os.mkdir(new_part_path)
        #     # Iou with all gts must below 0.25
        #     save_file = os.path.join(neg_save_dir + "/part%d" % part_n, "%s.jpg" % n_idx)
        #     f2.write("%s/negative/part%d/%s.jpg" % (net, part_n, n_idx) + ' 0 0 0 0 0 0\n')
        #     cv2.imwrite(save_file, resized_conds[neg_idx])
        #     n_idx += 1

    #f1.close()
    f2.close()
    #f3.close()


def test_net(dataset_path, annotation_file,  prefix, epoch, batch_size, test_mode="rnet",
             thresh=[0.6, 0.6, 0.7], min_face_size=24, stride=2, vis=False):

    detectors = [None, None, None]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    mtcnn_detector = JDAPDetector(detectors=detectors, min_face_size=min_face_size, stride=stride, threshold=thresh)

    # detect wider_face image
    fin = open(annotation_file, 'r')
    annots = fin.readlines()
    detections = []
    count = 0
    for line in annots:
        count += 1
        if count % 100 == 0:
            print("Handle image %d" % count)
        annot = line.strip().split(' ')
        im_path = os.path.join(dataset_path, annot[0])
        bbox = map(float, annot[1:])
        gt_boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        image = cv2.imread(im_path)
        boxes = mtcnn_detector.detect(image)

        # draw detection result
        if vis:
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                score = box[4]
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 0), 2)
            cv2.imshow('a', image)
            cv2.waitKey(0)
        detections.append(boxes)
    if test_mode == "pnet":
        net = "24"
    elif test_mode == "rnet":
        net = "48"

    save_path = os.path.join(data_dir, net)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, "detections_%s_0.4.pkl" % net)
    with open(save_file, 'wb') as f:
        cPickle.dump(detections, f, cPickle.HIGHEST_PROTOCOL)

# Ground Truth landmark points number
LANDMARK_POINTS = 5
SAMPLE_PER_IMAGE = 3
IOU_THRESH = 0.5
MAX_MEMORY = 3000 * 4000

def celeba_test_net_save(dataset_path, annotation_file, annotation_box_file, output_file, mtcnn_detector, vis=False):
    # detect celebA image
    fin_landmark = open(annotation_file, 'r')
    fin_box = open(annotation_box_file, 'r')
    fout = open(output_file, 'w')
    save_dir = os.path.join(data_dir, "%d/landmark" % netSize)
    annots_landmark = fin_landmark.readlines()
    annots_box = fin_box.readlines()
    count = 0
    landmark_idx = 0
    for line_id, (land_info, box_info) in enumerate(zip(annots_landmark, annots_box)):
        if line_id < 2:
            continue
        count += 1
        print("Handle image %d" % count)
        annot_land = land_info.strip().split()
        annot_box = box_info.strip().split()
        if annot_box[0] != annot_land[0]:
            continue
        im_path = os.path.join(dataset_path, annot_land[0])
        gt_landmarks = np.array(map(int, annot_land[1:]))
        gt_box = np.array(map(int, annot_box[1:]))
        # convert to x1 y1 x2 y2 mode
        gt_box[2] = gt_box[0] + gt_box[2]
        gt_box[3] = gt_box[1] + gt_box[3]
        image = cv2.imread(im_path)
        image_height, image_width, _ = image.shape
        # Avoid GPU memory broken
        if image_height * image_width >= MAX_MEMORY:
            continue
        boxes = mtcnn_detector.detect(image)
        ious = IoU(gt_box, boxes)
        if len(boxes) == 0 or len(gt_box) == 0:
            continue
        boxes = sorted(boxes[ious > IOU_THRESH], key=itemgetter(4), reverse=True)
        if len(boxes) == 0:
            continue
        boxes = np.array(boxes[:SAMPLE_PER_IMAGE])
        boxes_square = convert_to_square(boxes)
        boxes_square[:, 0:4] = np.round(boxes_square[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(boxes_square, image_width, image_height)
        for i, box in enumerate(boxes):
            reg_landmark = np.empty([LANDMARK_POINTS*2], dtype=np.float32)
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            box_w = x2 - x1
            box_h = y2 - y1
            reg_landmark[::2] = (gt_landmarks[::2] - x1) / box_w
            reg_landmark[1::2] = (gt_landmarks[1::2] - y1) / box_h
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            resized_img = cv2.resize(tmp, (netSize, netSize), interpolation=cv2.INTER_LINEAR)
            # Save landmark crop
            landmark_idx += 1
            save_file = os.path.join(save_dir, "%s.jpg" % landmark_idx)
            reg_value = ['%.2f' % t for t in reg_landmark]
            reg_value_str = ' '.join([t for t in reg_value])
            fout.write("%s/landmark/%s.jpg -2" % (netSize, landmark_idx) + ' ' + reg_value_str + '\n')
            cv2.imwrite(save_file, resized_img)

        # draw detection result
        if vis:
            for point_id in range(LANDMARK_POINTS):
                cv2.circle(image, (gt_landmarks[2 * point_id], gt_landmarks[2 * point_id + 1]), 4, (0, 200, 0), -1)
            cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 200), 2)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                score = box[4]
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 0), 2)
            cv2.imshow('a', image)
            cv2.waitKey(0)


def draw_rectangle(image, rect, color=(0,0,200)):
    rect = [int(x) for x in rect]
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)


def draw_text(image, point, text):
    if type(text) == list:
        s = ' '.join([str('%.2f' % x) for x in text])
    else:
        s = str(text)
    cv2.putText(image, s, point, 1, 1, (200, 0, 200))


def aflw_test_net_save(dataset_path, annotation_file, output_file, mtcnn_detector, vis=False):
    # detect AFLW image (filepath, rect[x,y,w,h], pose[p,y,r])
    fin = open(annotation_file, 'r')
    fout = open(output_file, 'w')
    save_dir = os.path.join(data_dir, "%d/pose" % netSize)
    annots = fin.readlines()
    count = 0
    pose_crop_id = 0
    for annot in annots:
        annot = annot.strip().split()
        fileName = annot[0]
        # Only one box
        gt_box = np.array(map(int, annot[1:5]))
        # Convert box mode
        gt_box[2] = gt_box[0] + gt_box[2]
        gt_box[3] = gt_box[1] + gt_box[3]
        gt_pose = np.array(map(float, annot[5:8]))
        count += 1
        print("AFLW %d image %s" % (count, fileName))
        image_path = os.path.join(dataset_path, fileName)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        if image_width * image_height >= MAX_MEMORY:
            continue
        boxes = mtcnn_detector.detect(image)
        if len(boxes) == 0 or len(gt_box) == 0:
            continue
        ious = IoU(gt_box, boxes)
        boxes = sorted(boxes[ious > IOU_THRESH], key=itemgetter(4), reverse=True)
        if len(boxes) == 0:
            continue
        boxes = np.array(boxes[:SAMPLE_PER_IMAGE])
        boxes_square = convert_to_square(boxes)
        boxes_square[:, 0:4] = np.round(boxes_square[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(boxes_square, image_width, image_height)
        for i, box in enumerate(boxes_square):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            resized_img = cv2.resize(tmp, (netSize, netSize), interpolation=cv2.INTER_LINEAR)
            gt_pose_str = ' '.join(['%.2f' % t_x for t_x in gt_pose])
            pose_crop_id += 1
            save_file = os.path.join(save_dir, "%s.jpg" % pose_crop_id)
            fout.write("%s/pose/%s.jpg -3" % (netSize, pose_crop_id) + ' ' + gt_pose_str + '\n')
            cv2.imwrite(save_file, resized_img)
            if vis:
                draw_rectangle(image, box, (200, 200, 0))

        if vis:
            draw_rectangle(image, gt_box)
            draw_text(image, (gt_box[0], gt_box[1]), gt_pose)
            cv2.imshow("a", image)
            cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        #default='/home/dafu/data/WIDER_FACE/WIDER_train/images'
                        #default='/home/dafu/data/CelebA/img_celeba'
                        default='/home/dafu/data/AFLW/data'
                        ,type=str)
    parser.add_argument('--image_set', dest='image_set', help='image set',
                        default='wider_face_train', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='rnet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['../models/pnet/pnet_wider_OHEM_0.7/pnet',
                                 '../models/rnet/rnet_wider_OHEM_0.7/rnet',
                                 ''], type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[16, 16, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.4, 0.1, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection', default=24, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window', default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with', default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', default=False, help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args

    stageName = 'pose'
    annotation_file = '/home/dafu/data/WIDER_FACE/wider_face_split/wider_train_annotation.txt'
    # Load model
    mtcnn_detector = load_model(args.prefix, args.epoch, args.batch_size, args.test_mode, args.thresh, args.min_face)

    # Crop and Save head and face samples
    #expandScale = 0.3
    #save_head_face_example(24, expandScale)

    # test_net(args.dataset_path, annotation_file, args.prefix,
    #          args.epoch, args.batch_size, args.test_mode,
    #          args.thresh, args.min_face, args.stride, args.vis)
    #save_hard_example("48")

    if stageName == 'landmark':
        # Landmark samples
        annotation_file = '/home/dafu/data/CelebA/list_landmarks_celeba.txt'
        annotation_box_file = '/home/dafu/data/CelebA/list_bbox_celeba.txt'
        output_file = '/home/dafu/data/jdap_data/48/train_landmark_48.txt'
        celeba_test_net_save(args.dataset_path, annotation_file, annotation_box_file, output_file, mtcnn_detector, args.vis)
    elif stageName == 'pose':
        # Pose samples
        annotation_file = '/home/dafu/data/AFLW/data/aflw_rect_pose.txt'
        output_file = '/home/dafu/data/jdap_data/48/train_pose_48.txt'
        aflw_test_net_save(args.dataset_path, annotation_file, output_file, mtcnn_detector, args.vis)