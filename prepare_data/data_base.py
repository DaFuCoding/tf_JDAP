import os.path as osp
import numpy as np
from collections import OrderedDict
from tools.utils import *
import cv2

__all__ = ['FDDB', 'WIDER', 'CelebA', 'AFLW', 'L300WP', 'LS3DW']


def redetect(image_info, detector):
    image = cv2.imread(image_info['image_name'])
    height, width, _ = image.shape
    gt_box = image_info['gt_boxes']
    w = gt_box[2] - gt_box[0] + 1
    h = gt_box[3] - gt_box[1] + 1
    max_side = max(h, w) * 1.2
    square_bbox = gt_box.copy()
    square_bbox[0] = gt_box[0] + w * 0.5 - max_side * 0.5
    square_bbox[1] = gt_box[1] + h * 0.5 - max_side * 0.5
    square_bbox[2] = square_bbox[0] + max_side - 1
    square_bbox[3] = square_bbox[1] + max_side - 1
    square_bbox = np.array([int(x) for x in square_bbox])
    dets = square_bbox[np.newaxis, :]

    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, width, height)
    cropped_ims = np.zeros((1, 48, 48, 3), dtype=np.float32)
    for i in range(1):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        cropped_ims[i, :, :, :] = resize_image_by_wh(tmp, (48, 48))
    results = detector.predict(cropped_ims)

    head_pose = results[2]
    landmark_reg = np.reshape(results[-1][0], [-1, 2])
    # print(head_pose * 180 / 3.14)
    land_point = np.transpose(
        np.vstack((square_bbox[0] + landmark_reg[:, 0].T * max_side,
                   square_bbox[1] + landmark_reg[:, 1].T * max_side))).astype(np.int32)
    # gt_landmark = image_info['landmarks']
    # for l in range(68):
    #     cv2.circle(image, (gt_landmark[2 * l], gt_landmark[2 * l + 1]), 1, (200, 0, 0), 2)
    #     cv2.circle(image, (land_point[l][0], land_point[l][1]), 1, (0, 0, 200), 2)
    # cv2.imshow('a', image)
    # cv2.waitKey(0)
    land_point = np.reshape(land_point, [1, -1])
    return head_pose, land_point


class _DataBase(object):
    def __init__(self, dataset_path, image_name_file):
        self.label_infos = self.get_labels(image_name_file)
        self.dataset_path = dataset_path

    def get_labels(self, image_name_file):
        if not osp.exists(image_name_file):
            raise ValueError("File is not exist.")
        with open(image_name_file, 'r') as f:
            lines = f.readlines()
        return lines

    def get_image_name(self, label_info):
        image_name = label_info.strip().split()[0]
        return osp.join(self.dataset_path, image_name)

    def eval_accuracy(self, eval_file, eval_dim=7, eval_mode='landmark_pose-task', detector=None, vis=False):
        """

        Args:
            eval_file:
            eval_dim:
            eval_mode:
            detector: if fail detect face, using detector
            vis:

        Returns:

        """
        fail_image_info = list()
        seven_points_idx = [36, 39, 42, 45, 30, 48, 54]
        # [0, 30] (30, 60] (60, ]
        yaw_pose_cls_error_pose = np.zeros([3, 3], dtype=np.float32)
        yaw_pose_cls_error_landmark = np.zeros([3, eval_dim], dtype=np.float32)
        yaw_pose_cls_sample_num = np.zeros([3], dtype=np.int32)
        with open(eval_file, 'r') as f:
            detect_result = f.readlines()
            valid_num = 0
            for gt_info, pred in zip(self.label_infos, detect_result):
                gt_result = self.label_parser(gt_info)
                gt_image_name = gt_result['image_name'][len(self.dataset_path) + 1:].strip()
                gt_box = gt_result['gt_boxes']
                gt_pose = gt_result['head_pose']
                yaw_pose_cls_index = np.abs(int(gt_pose[1] * 180 / 3.14)) / 30
                if yaw_pose_cls_index >= 3:
                    yaw_pose_cls_index = 2
                gt_landmark = np.reshape(gt_result['landmarks'], [-1, 2]).astype(np.int32)
                normalizer = np.sqrt((gt_box[3] - gt_box[1] + 1) * (gt_box[2] - gt_box[0] + 1))

                detect_info = pred.strip().split()
                image_name = detect_info[0]
                if image_name != gt_image_name:
                    assert (image_name + ' is not match ' + gt_image_name)
                cand_num = int(detect_info[1])
                pred_info = np.array(detect_info[2:])
                box_pred = np.reshape(pred_info[:cand_num * 4], [cand_num, 4]).astype(np.int32)
                pred_info = pred_info[cand_num * 4:]
                is_fail_detect = False
                if cand_num == 0:
                    #print(image_name + ' fail detect.')
                    fail_image_info.append(gt_result)
                    is_fail_detect = True
                    best_id = 0
                    pose_pred, landmark_pred = redetect(gt_result, detector)
                else:
                    ious = IoU(gt_box, box_pred)
                    best_id = np.argmax(ious)
                    if ious[best_id] < 0.4:
                        #print(gt_image_name + ' without suitable face.')
                        fail_image_info.append(gt_result)
                        is_fail_detect = True
                        best_id = 0
                        pose_pred, landmark_pred = redetect(gt_result, detector)

                if is_fail_detect == False:
                    if 'pose' in eval_mode:
                        pose_pred = np.reshape(pred_info[:cand_num * 3], [cand_num, 3]).astype(np.float32)
                        pred_info = pred_info[cand_num * 3:]
                    if 'landmark' in eval_mode:
                        landmark_pred = np.reshape(pred_info, [cand_num, -1]).astype(np.int32)

                if 'landmark' in eval_mode:
                    landmark_pred = np.reshape(np.array(landmark_pred[best_id], np.int32), [-1, 2])
                    if eval_dim == 7:
                        gt_landmark = gt_landmark[seven_points_idx]
                        if landmark_pred.shape[0] != 7:
                            landmark_pred = landmark_pred[seven_points_idx]
                    elif eval_dim == 68:
                        assert (landmark_pred.shape[0] == eval_dim)

                    yaw_pose_cls_error_landmark[yaw_pose_cls_index] += \
                        np.sqrt(np.square(gt_landmark[:, 0] - landmark_pred[:, 0]) +
                                np.square(gt_landmark[:, 1] - landmark_pred[:, 1])) / normalizer
                if 'pose' in eval_mode:
                    pose_pred = pose_pred[best_id]
                    yaw_pose_cls_error_pose[yaw_pose_cls_index] += np.abs(gt_pose - pose_pred)
                valid_num += 1
                yaw_pose_cls_sample_num[yaw_pose_cls_index] += 1
                if vis and is_fail_detect:
                    image = cv2.imread(gt_result['image_name'])
                    for i in range(eval_dim):
                        cv2.circle(image, (gt_landmark[i][0], gt_landmark[i][1]), 1, (0, 0, 255), 2)
                    for i in range(eval_dim):
                        cv2.circle(image, (int(landmark_pred[i][0]), int(landmark_pred[i][1])), 1, (0, 255, 0), 2)
                    for m in range(cand_num):
                        cv2.rectangle(image, (box_pred[m][0], box_pred[m][1]), (box_pred[m][2], box_pred[m][3]),
                                      (0, 0, 128), 2)
                    cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (200, 200, 0), 2)
                    cv2.imshow('a', image)
                    cv2.waitKey(0)

            print("angle bin: [0,30]   (30,60]    (60,)")
            print("sample num:")
            print(yaw_pose_cls_sample_num)
            if 'landmark' in eval_mode:
                #print(yaw_pose_cls_error_landmark / yaw_pose_cls_sample_num)
                print("small    medium  large")
                print(np.sum(np.transpose(yaw_pose_cls_error_landmark) / yaw_pose_cls_sample_num, axis=0))
            if 'pose' in eval_mode:
                print(np.transpose((np.transpose(yaw_pose_cls_error_pose) / yaw_pose_cls_sample_num) * 180 / 3.14))
        results = {'fail_image_info': fail_image_info, 'yaw_pose_cls_sample_num': yaw_pose_cls_sample_num,
                   'yaw_pose_cls_error_pose': yaw_pose_cls_error_pose,
                   'yaw_pose_cls_error_landmark': yaw_pose_cls_error_landmark}
        return results


class FDDB(_DataBase):
    def __init__(self, dataset_path, image_name_file):
        super(FDDB, self).__init__(dataset_path, image_name_file)

    # TODO: parser regular
    def label_parser(self, label_info):
        info = label_info.strip().split()
        image_name = info[0]
        if '.jpg' not in image_name:
            image_name += '.jpg'
        return osp.join(self.dataset_path, image_name)

    def get_image_name(self, label_info, Full=1):
        image_name = label_info.strip().split()[0]
        if '.jpg' not in image_name:
            image_name += '.jpg'
        return osp.join(self.dataset_path, image_name)

    def do_eval(self, image_name, results):
        #pure_image_name = os.path.splitext(image_name.split(self.dataset_path)[-1][1:])
        pure_image_name = image_name.strip()
        cal_boxes = results[0] if type(results) is tuple else results

        box_num = cal_boxes.shape[0]
        x = cal_boxes[:, 0]
        y = cal_boxes[:, 1]
        w = cal_boxes[:, 2] - cal_boxes[:, 0] + 1
        h = cal_boxes[:, 3] - cal_boxes[:, 1] + 1
        scores = cal_boxes[:, 4]

        write_result = '%s \n%d\n' % (pure_image_name, box_num)
        for i in range(box_num):
            # Face rect and score
            write_result += '%d %d %d %d %.4f\n' % (x[i], y[i], w[i], h[i], scores[i])
        return write_result


class WIDER(_DataBase):
    def __init__(self, mode):
        # User custom
        dataset_root_path = '/home/dafu/data/WIDER_FACE'
        if mode == 'train':
            image_name_file = 'wider_face_split/wider_train_annotation.txt'
            image_dir_name = 'WIDER_train/images'
        elif mode == 'val':
            image_name_file = 'wider_face_split/wider_val_annotation.txt'
            image_dir_name = 'WIDER_val/images'
        else:
            raise Exception("mode use 'train' or 'val'.")
        image_name_file = osp.join(dataset_root_path, image_name_file)
        super(WIDER, self).__init__(dataset_root_path, image_name_file)
        self.dataset_image_path = osp.join(dataset_root_path, image_dir_name)
        self.mode = mode


    # TODO: parser regular
    def label_parser(self, label_info):
        label_info = label_info.strip().split(' ')
        image_name = osp.join(self.dataset_image_path, label_info[0])
        boxes = np.array(map(float, label_info[1:]), dtype=np.float32).reshape(-1, 4)
        return OrderedDict({'image_name': image_name, 'gt_boxes': boxes})

    def get_image_name(self, label_info):
        image_name = label_info.strip().split()[0]
        return osp.join(self.dataset_image_path, image_name)


class L300WP(_DataBase):
    # http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
    # The synthesized large-pose face images from 300W.
    # Annotation:
    # image_name tight_rect head_pose landmark
    # tight_rect(int): x1 y1 x2 y2
    # head_pose(radian float): pitch(up+) yaw(right+) roll(left+)
    # landmark(int): eyes corner, nose, mouth cornet index [36, 39, 42, 45, 30, 48, 54]
    def __init__(self, dataset_path, image_name_file, mode='train'):
        super(L300WP, self).__init__(dataset_path, image_name_file)
        self.mode = mode

    def get_keys(self):
        return ['image_name', 'gt_boxes', 'head_pose', 'landmarks']

    def label_parser(self, label_info):
        label_info = label_info.strip().split(' ')
        image_name = osp.join(self.dataset_path, label_info[0])
        boxes = np.array(label_info[1:5], dtype=np.int32)
        head_pose = np.array(label_info[5:8], dtype=np.float32)
        landmarks = np.array(label_info[8:], dtype=np.int32)
        return OrderedDict({'image_name': image_name, 'gt_boxes': boxes,
                            'head_pose': head_pose, 'landmarks': landmarks})

    def do_eval(self, label_info, results, task='landmark_pose'):
        info_dict = self.label_parser(label_info)
        cal_boxes = results[0] if type(results) is tuple else results
        pure_name = (info_dict['image_name']).split('/')[-1]
        write_result = pure_name
        if (cal_boxes is None) or len(cal_boxes) == 0:
            print(info_dict['image_name'] + ' fail to detect.')
            return write_result + ' 0 \n'

        best_box = np.round(cal_boxes).astype(np.int32)
        has_landmark = False
        if 'landmark' in task:
            has_landmark = True
            landmark_reg = results[-1]
        has_head_pose = False
        if 'pose' in task:
            has_head_pose = True
            pose_reg = results[1]
        cand_box_num = len(best_box)
        write_result += ' %d ' % cand_box_num
        for box_id in range(cand_box_num):
            write_result += ''.join(['%d ' % x for x in best_box[box_id][:-1]])
        if has_head_pose:
            for box_id in range(cand_box_num):
                write_result += '%.4f %.4f %.4f ' % (pose_reg[box_id][0], pose_reg[box_id][1], pose_reg[box_id][2])
        if has_landmark:
            landmard_pt_num = landmark_reg.shape[-1] // 2
            for box_id in range(cand_box_num):
                for l in range(landmard_pt_num):
                    write_result += '%d %d ' % (landmark_reg[box_id][l], landmark_reg[box_id][landmard_pt_num + l])
        write_result += '\n'
        return write_result


class LS3DW(_DataBase):
    def __init__(self, dataset_path, image_name_file, mode='train'):
        super(LS3DW, self).__init__(dataset_path, image_name_file)
        self.mode = mode

    def label_parser(self, label_info):
        label_info = label_info.strip().split(' ')
        image_name = osp.join(self.dataset_path, label_info[0])
        boxes = np.array(label_info[1:5], dtype=np.int32)
        landmarks = np.array(label_info[5:], dtype=np.int32)
        return OrderedDict({'image_name': image_name, 'gt_boxes': boxes, 'landmarks': landmarks})

    def do_eval(self, label_info, results, task='landmark_pose'):
        info_dict = self.label_parser(label_info)
        cal_boxes = results[0] if type(results) is tuple else results
        pure_name = (info_dict['image_name']).split('/')[-1]
        write_result = pure_name
        if (cal_boxes is None) or len(cal_boxes) == 0:
            print(info_dict['image_name'] + ' fail to detect.')
            return write_result + ' 0 \n'

        best_box = np.round(cal_boxes).astype(np.int32)
        landmark_reg = results[-1]
        cand_box_num = len(best_box)
        write_result += ' %d ' % cand_box_num
        for box_id in range(cand_box_num):
            write_result += '%d %d %d %d ' % (best_box[box_id][0], best_box[box_id][1], best_box[box_id][2], best_box[box_id][3])
        if 'landmark' in task:
            for box_id in range(cand_box_num):
                for l in range(landmark_reg.shape[-1]//2):
                    write_result += '%d %d ' % (landmark_reg[box_id][l], landmark_reg[box_id][landmark_reg.shape[-1]//2 + l])
        write_result += '\n'
        return write_result


class CelebA(_DataBase):
    pass

class AFLW(_DataBase):
    pass
