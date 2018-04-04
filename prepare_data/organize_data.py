#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np
import numpy.random as npr
import os
from configs.cfg import config
from data_base import WIDER
from tools.utils import *

__all__ = ['DataPretreat']

class GenerateRect(object):
    """
        Generate rectangle by different methods
    """
    def __init__(self, box):
        pass
    @staticmethod
    def random_rect(height, width, net_size):
        size = npr.randint(net_size, min(height, width) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        return np.array([nx, ny, nx + size, ny + size])

    @staticmethod
    def random_jitter():
        size = npr.randint(12, min(width, height) / 2)
        # delta_x and delta_y are offsets of (x1, y1)
        delta_x = npr.randint(max(-size, -x1), w)
        delta_y = npr.randint(max(-size, -y1), h)
        nx1 = int(max(0, x1 + delta_x))
        ny1 = int(max(0, y1 + delta_y))

    @staticmethod
    def random_center_jitter(h, w):
        size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

        # delta here is the offset of box center
        delta_x = npr.randint(-w * 0.2, w * 0.2)
        delta_y = npr.randint(-h * 0.2, h * 0.2)

        nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
        ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
        nx2 = int(nx1 + size)
        ny2 = int(ny1 + size)

    @staticmethod
    def rect_to_square(rect):
        x1, y1 = rect[0], rect[1]
        w = rect[2] - rect[0] + 1
        h = rect[3] - rect[1] + 1
        best_side = int(np.sqrt(w * h))
        best_x1 = int(max(x1 + (w - best_side) / 2, 0))
        best_y1 = int(max(y1 + (h - best_side) / 2, 0))
        best_x2 = int(best_x1 + best_side - 1)
        best_y2 = int(best_y1 + best_side - 1)
        return np.array([best_x1, best_y1, best_x2, best_y2])


class GenerateData(object):
    """
        Main generate three mode data: positive, negative and part sample
    """
    def __init__(self, db_indicator, self_data_root_path, config, extra_name=''):
        self.db_indicator = db_indicator
        self.mode = self.db_indicator.mode
        self.self_data_root_path = self_data_root_path
        self.extra_name = extra_name
        self.prefix_name = '%s_%s' % (self.mode, self.extra_name)
        self.config = config
        self.net_size = self.config['net_size']
        self.net_root_path = opj(self.self_data_root_path, str(self.net_size))
        self.annotations = db_indicator.label_infos
        self.num_of_images = len(self.annotations)


    def folder_check_make(self):
        self.neg_save_dir = opj(self.net_root_path, "%snegative" % self.prefix_name)
        self.pos_save_dir = opj(self.net_root_path, "%spositive" % self.prefix_name)
        self.part_save_dir = opj(self.net_root_path, "%spart" % self.prefix_name)
        for dir_path in [self.neg_save_dir, self.pos_save_dir, self.part_save_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def file_check_open(self):
        self.fpos = open(opj(self.net_root_path, '%spos_%d.txt' % (self.prefix_name, self.net_size)), 'w')
        self.fneg = open(opj(self.net_root_path, '%sneg_%d.txt' % (self.prefix_name, self.net_size)), 'w')
        self.fpart = open(opj(self.net_root_path, '%spart_%d.txt' % (self.prefix_name, self.net_size)), 'w')

    def image_resize(self, image):
        return cv2.resize(image, (self.net_size, self.net_size), interpolation=dp.random_resize_method)

    def _generate_background_rect(self, height, width, gt_boxes):
        inside_neg_num = 0
        outside_neg_num = 0

        bkg_boxes = list()

        # bkg over with gt
        while outside_neg_num < self.config.neg_outside_num_per_image:
            size = npr.randint(self.net_size, min(width, height) // 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            bkg_box = np.array([nx, ny, nx + size, ny + size])
            if np.max(IoU(bkg_box, gt_boxes)) < self.config.neg_iou:
                # Iou with all gts must below 0.3
                bkg_boxes.append(bkg_box)
                outside_neg_num += 1

        # bkg overlap with gt
        while inside_neg_num < self.config.neg_inside_num_per_image:
            size = npr.randint(self.net_size, min(width, height) // 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            bkg_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            if np.max(IoU(bkg_box, gt_boxes)) < config.neg_iou:
                bkg_boxes.append(bkg_box)
                inside_neg_num += 1


        return np.array(bkg_boxes)

    def _generate_foreground_rect(self, height, width, gt_boxes):
        pass


    def generate(self):
        for annotation in annotations:
            annotations = dataset_indicator.label_infos
            num_of_images = len(annotations)
            print ("processing %d images in total" % num_of_images)
            img = cv2.imread(im_path)
            idx += 1
            if idx % 100 == 0:
                print(idx, "images done")

            height, width, channel = img.shape
            neg_num = 0
            while neg_num < config.neg_outside_num_per_image:
                crop_box = self.random_rect()
                iou = IoU(crop_box, boxes)

                cropped_im = img[ny: ny + size, nx: nx + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=dp.random_resize_method)

                if np.max(Iou) < config.neg_iou:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write("12/negative/%s.jpg" % n_idx + ' 0 0 0 0 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                    neg_num += 1

    def __del__(self):
        self.fpos.close()
        self.fneg.close()
        self.fpart.close()


class DataPretreat(object):
    """
        Data color augment, reference to Caffe2 "image_input_op.h"
    """
    def __init__(self):
        # Resize method
        self.resize_method = [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR]
        # Data color augment parameter
        self.saturation = 0.4
        self.brightness = 0.4
        self.contrast = 0.4

    @property
    def random_resize_method(self):
        #select_method = npr.random(3).argmax()
        #return self.resize_method[select_method]
        return self.resize_method[2]

    def uchar_protect(self, array):
        over_idx = np.where(array > 255)
        low_idx = np.where(array < 0)
        array[over_idx] = 255
        array[low_idx] = 0
        return array

    def to_grey(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, axis=2)
        return image

    def Brightness(self, image, alpha_rand=0.4):
        alpha = 1.0 + npr.uniform(-alpha_rand, alpha_rand)
        image = self.uchar_protect(image * alpha)
        return image.astype(np.uint8)


    def Contrast(self, image, alpha_rand=0.4):
        h, w, c = image.shape
        gray_mean = np.sum(0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2])
        gray_mean /= h * w
        alpha = 1.0 + npr.uniform(-alpha_rand, alpha_rand)
        image = self.uchar_protect(image * alpha + gray_mean * (1.0 - alpha))
        return image.astype(np.uint8)


    def Saturation(self, image, alpha_rand=0.4):
        alpha = 1.0 + npr.uniform(-alpha_rand, alpha_rand)
        gray_color = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        gray_color = gray_color[:, :, np.newaxis]
        gray_color = np.repeat(gray_color, 3, axis=2)
        image = self.uchar_protect(image * alpha + gray_color * (1.0 - alpha))
        return image.astype(np.uint8)

    def ColorJitter(self, image, saturation=0.4, brightness=0.4, contrast=0.4):
        jitter_order = [0, 1, 2]
        npr.shuffle(jitter_order)
        for i in range(3):
            if jitter_order[i] == 0:
                image = self.Saturation(image, saturation)
            elif jitter_order[i] == 1:
                image = self.Brightness(image, brightness)
            else:
                image = self.Contrast(image, contrast)
        return image


# if __name__ == '__main__':
#     db_indicator = WIDER(config.gl.wider_face_root_path, '', 'train')
#
#     data_generator = GenerateData(db_indicator, config.gl.self_data_root_path, config.pnet)
