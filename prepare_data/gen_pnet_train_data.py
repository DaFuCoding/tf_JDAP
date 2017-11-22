#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import os
import numpy.random as npr
from easydict import EasyDict as edict
import sys
import os.path as osp
sys.path.append(osp.join('.'))
from tools.utils import *


# PNet config
config = edict()
config.pos_iou = 0.65
config.part_iou = 0.4
config.neg_iou = 0.3

config.neg_outside_num_per_image = 50
config.neg_inside_num_per_image = 5  # 50 + 5 : 80W negative samples
# 20 : 18W positive and 52W part samples
# 25 : 27W positive and 65W part samples
config.pos_num_per_image = 25
# color jitter : grey : original image = 2 : 1 : 2
#config.pos_aug_ratio = [0.4, 0.6, 1.0]
# Without data augment
config.pos_aug_ratio = [0., 0., 1.0]
config.min_face_size = 40

net_size = 12

# WIDER-FACE image and annotation path
data_root_dir = '/home/dafu/data/WIDER_FACE'
anno_file = os.path.join('./prepare_data/wider_train_annotation.txt')

image_root_dir = os.path.join(data_root_dir, 'WIDER_train/images')
# Save patch image path
save_dir_root = '/home/dafu/data/jdap_data'
neg_save_dir = os.path.join(save_dir_root, "%d/negative" % net_size)
pos_save_dir = os.path.join(save_dir_root, "%d/positive" % net_size)
part_save_dir = os.path.join(save_dir_root, "%d/part" % net_size)
save_dir = os.path.join(save_dir_root, "%d" % net_size)
for dir_path in [save_dir, neg_save_dir, pos_save_dir, part_save_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
# Create output file
f1 = open(os.path.join(save_dir, 'pos_%d.txt' % net_size), 'w')
f2 = open(os.path.join(save_dir, 'neg_%d.txt' % net_size), 'w')
f3 = open(os.path.join(save_dir, 'part_%d.txt' % net_size), 'w')

with open(anno_file, 'r') as f:
    annotations = f.readlines()

num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

dp = DataPretreat()

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = os.path.join(image_root_dir, annotation[0])
    bbox = map(float, annotation[1:])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")

    height, width, channel = img.shape
    neg_num = 0
    while neg_num < config.neg_outside_num_per_image:
        size = npr.randint(12, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])
        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=dp.random_resize_method)

        if np.max(Iou) < config.neg_iou:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write("12/negative/%s.jpg" % n_idx + ' 0 0 0 0 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < config.min_face_size or x1 < 0 or y1 < 0:
            continue

        # Add ground truth samples
        best_side = int(np.sqrt(w * h))
        best_x1 = int(max(x1 + (w - best_side) / 2, 0))
        best_y1 = int(max(y1 + (h - best_side) / 2, 0))
        best_x2 = int(best_x1 + best_side - 1)
        best_y2 = int(best_y1 + best_side - 1)
        if best_x2 <= width and best_y2 <= height:
            gt_img = img[best_y1:best_y2, best_x1:best_x2, :]
            resized_im = cv2.resize(gt_img, (12, 12), interpolation=dp.random_resize_method)
            offset_x1 = (x1 - best_x1) / float(best_side)
            offset_y1 = (y1 - best_y1) / float(best_side)
            offset_x2 = (x2 - best_x2) / float(best_side)
            offset_y2 = (y2 - best_y2) / float(best_side)
            save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
            f1.write("12/positive/%s.jpg" % p_idx + ' 1 %.2f %.2f %.2f %.2f\n'
                     % (offset_x1, offset_y1, offset_x2, offset_y2))
            cv2.imwrite(save_file, resized_im)
            p_idx += 1

        # generate negative examples that have overlap with gt
        for _ in range(config.neg_inside_num_per_image):
            size = npr.randint(12,  min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=dp.random_resize_method)

            if np.max(Iou) < config.neg_iou:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("12/negative/%s.jpg" % n_idx + ' 0 0 0 0 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # generate positive examples and part faces
        for i in range(config.pos_num_per_image):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = int(nx1 + size)
            ny2 = int(ny1 + size)

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=dp.random_resize_method)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= config.pos_iou:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write("12/positive/%s.jpg" % p_idx + ' 1 %.2f %.2f %.2f %.2f\n'
                         % (offset_x1, offset_y1, offset_x2, offset_y2))
                aug_image = resized_im
                aug_type = npr.random(1)
                if aug_type < config.pos_aug_ratio[0]:
                    aug_image = dp.ColorJitter(aug_image)
                elif aug_type < config.pos_aug_ratio[1]:
                    aug_image = dp.to_grey(aug_image)
                cv2.imwrite(save_file, aug_image)
                p_idx += 1
            elif IoU(crop_box, box_) >= config.part_iou:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("12/part/%s.jpg" % d_idx + ' -1 %.2f %.2f %.2f %.2f\n'
                         % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
