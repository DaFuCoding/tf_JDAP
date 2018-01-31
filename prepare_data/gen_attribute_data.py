from demo.detectAPI import DetectAPI
from data_base import *
from tools.utils import *
import os
import cv2
import numpy as np
from operator import itemgetter, attrgetter

# Ground Truth landmark points number
LANDMARK_POINTS = 68
SAMPLE_PER_IMAGE = 3
IOU_THRESH = 0.65
MAX_MEMORY = 3000 * 4000
net_size = 48

def celeba_test_net_save(dataset_path, annotation_file, annotation_box_file, output_file, mtcnn_detector, vis=False):
    # detect celebA image
    fin_landmark = open(annotation_file, 'r')
    fin_box = open(annotation_box_file, 'r')
    fout = open(output_file, 'w')
    save_dir = os.path.join(data_dir, "%d/landmark" % net_size)
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
            resized_img = cv2.resize(tmp, (net_size, net_size), interpolation=cv2.INTER_LINEAR)
            # Save landmark crop
            landmark_idx += 1
            save_file = os.path.join(save_dir, "%s.jpg" % landmark_idx)
            reg_value = ['%.2f' % t for t in reg_landmark]
            reg_value_str = ' '.join([t for t in reg_value])
            fout.write("%s/landmark/%s.jpg -2" % (net_size, landmark_idx) + ' ' + reg_value_str + '\n')
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


def aflw_test_net_save(dataset_path, annotation_file, output_file, mtcnn_detector, vis=False):
    # detect AFLW image (filepath, rect[x,y,w,h], pose[p,y,r])
    fin = open(annotation_file, 'r')
    fout = open(output_file, 'w')
    save_dir = os.path.join(data_dir, "%d/pose" % net_size)
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
            resized_img = cv2.resize(tmp, (net_size, net_size), interpolation=cv2.INTER_LINEAR)
            gt_pose_str = ' '.join(['%.2f' % t_x for t_x in gt_pose])
            pose_crop_id += 1
            save_file = os.path.join(save_dir, "%s.jpg" % pose_crop_id)
            fout.write("%s/pose/%s.jpg -3" % (net_size, pose_crop_id) + ' ' + gt_pose_str + '\n')
            cv2.imwrite(save_file, resized_img)
            if vis:
                draw_rectangle(image, box, (200, 200, 0))

        if vis:
            draw_rectangle(image, gt_box)
            draw_text(image, (gt_box[0], gt_box[1]), gt_pose)
            cv2.imshow("a", image)
            cv2.waitKey(0)

def convert_to_wider_face(gt_box, scale):
    w = gt_box[2] - gt_box[0] + 1
    h = gt_box[3] - gt_box[1] + 1
    gt_box[1] -= max(w, h) * scale
    return gt_box
import matplotlib.pyplot as plt

def test_core(detector, dataset_indicator, save_dir, output_file, vis=False):
    attrib_names = dataset_indicator.get_keys()
    patch_id = 0
    image_idx = 0
    # mean shape in all candidate boxes of train set
    mean_shape = np.zeros([136], dtype=np.float32)
    if vis is False:
        fout = open(output_file, 'w')
    for label_info in dataset_indicator.label_infos:
        image_idx += 1
        data_dict = dataset_indicator.label_parser(label_info)
        image_name = data_dict[attrib_names[0]]
        gt_box = data_dict[attrib_names[1]]
        gt_box = convert_to_wider_face(gt_box, 0.2)
        head_pose = data_dict[attrib_names[2]]
        gt_landmarks = data_dict[attrib_names[3]]
        print(image_name)
        image = cv2.imread(image_name)
        image_height, image_width, _ = image.shape
        # Avoid GPU memory broken
        if image_height * image_width >= MAX_MEMORY:
            continue
        # Detect image
        bbox_c = detector.detect(image)
        if len(bbox_c) == 0 or len(gt_box) == 0:
            continue
        boxes_square = convert_to_square(bbox_c)
        boxes_square[:, 0:4] = np.round(boxes_square[:, 0:4])
        ious = IoU(gt_box, boxes_square)
        boxes_square = sorted(boxes_square[ious > IOU_THRESH], key=itemgetter(4), reverse=True)
        if len(boxes_square) == 0:
            continue
        boxes_square = np.array(boxes_square[:SAMPLE_PER_IMAGE])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(boxes_square, image_width, image_height)

        for i, det_box in enumerate(boxes_square):
            reg_landmark = np.empty([LANDMARK_POINTS * 2], dtype=np.float32)
            x1, y1, x2, y2 = [box for box in det_box[:-1]]
            box_w = x2 - x1 + 1
            box_h = y2 - y1 + 1
            reg_landmark[::2] = (gt_landmarks[::2] - x1) / box_w
            reg_landmark[1::2] = (gt_landmarks[1::2] - y1) / box_h
            mean_shape += reg_landmark
            patch_id += 1
            # x = reg_landmark[0::2]
            # y = reg_landmark[1::2]
            # plt.plot(x, y, 'b.')
            # plt.show()
            if vis is False:
                # Crop image patch
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                resized_img = cv2.resize(tmp, (net_size, net_size), interpolation=cv2.INTER_LINEAR)

                # Save cropped image
                save_file = os.path.join(save_dir, "attribute/%d.jpg" % patch_id)
                reg_value = ['%.4f' % t for t in reg_landmark]
                head_pose_str = ' '.join(['%.4f' % t for t in head_pose])
                reg_value_str = ' '.join([t for t in reg_value])
                write_str = ' '.join(["%d/attribute/%d.jpg -4" % (net_size, patch_id), head_pose_str, reg_value_str])
                fout.write(write_str + '\n')
                cv2.imwrite(save_file, resized_img)

        # draw detection result
        if False:
            for point_id in range(LANDMARK_POINTS):
                cv2.circle(image, (gt_landmarks[2 * point_id], gt_landmarks[2 * point_id + 1]), 1, (0, 200, 0), -1)
            cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 200), 2)
            for box in boxes_square:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                score = box[4]
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 0), 1)
            cv2.imshow('a', image)
            cv2.waitKey(0)

    mean_shape = mean_shape / patch_id
    np.save('300WLP_mean_shape.npy', mean_shape)


if __name__ == '__main__':

    # Load model and dataset_indicator
    detector = DetectAPI(['../models/pnet/pnet_OHEM_0.7_wo_pooling/pnet',
                          '../models/rnet/rnet_wider_OHEM_0.7_wop_pnet/rnet', ''],
                         [13, 16, 16], "rnet", [2048, 256, 16], False, [0.4, 0.1, 0.1], 64)
    dataset_name = '300WLP'
    if dataset_name == '300WLP':
        dataset_dir = '/home/dafu/data/300W-LP'
        save_dir = '/home/dafu/data/jdap_data/%d' % net_size
        output_file = os.path.join(save_dir, '300WLP_attribute_test.txt')
        dataset_indicator = L300WP(dataset_dir, os.path.join(dataset_dir, '300WLP_rect_pose_landmark_68.txt'), 'TRAIN')
        test_core(detector, dataset_indicator, save_dir, output_file, vis=True)


    # if stageName == 'landmark':
    #     # Landmark samples
    #     annotation_file = '/home/dafu/data/CelebA/list_landmarks_celeba.txt'
    #     annotation_box_file = '/home/dafu/data/CelebA/list_bbox_celeba.txt'
    #     output_file = '/home/dafu/data/jdap_data/48/train_landmark_48.txt'
    #     celeba_test_net_save(args.dataset_path, annotation_file, annotation_box_file, output_file, mtcnn_detector, args.vis)
    # elif stageName == 'pose':
    #     # Pose samples
    #     annotation_file = '/home/dafu/data/AFLW/data/aflw_rect_pose.txt'
    #     output_file = '/home/dafu/data/jdap_data/48/train_pose_48.txt'
    #     aflw_test_net_save(args.dataset_path, annotation_file, output_file, mtcnn_detector, args.vis)