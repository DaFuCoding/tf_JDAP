# _*_ coding: utf-8 _*_
import os

import numpy as np
import cv2

from nets.JDAP_Net import JDAP_12Net_wo_pooling as P_Net
from nets.JDAP_Net import JDAP_24Net as R_Net
from nets.JDAP_Net import JDAP_mNet as M_Net
from nets.JDAP_Net import JDAP_24Net_ERC as R_Net_ERC
from nets.JDAP_Net import JDAP_48Net as O_Net
from nets.JDAP_Net import JDAP_48Net_Lanmark_Pose as O_AUX_Net
from fcn_detector import FcnDetector
from jdap_detect import JDAPDetector
from detector import Detector


class DetectAPI(object):
    def __init__(self, prefix, epoch, test_mode, batch_size, is_ERC, thresh, min_face_size):
        # To do compare experiments
        self.fix_param = [test_mode, batch_size, is_ERC, thresh, min_face_size]
        self.jdap_detector = self._init_model(prefix, epoch)

    # TODO:Loading new model, other no change
    def reset_model(self, prefix, epoch):
        # load new model
        # replace self.jdap_detector
        pass

    def _init_model(self, prefix, epoch):
        test_mode, batch_size, is_ERC, thresh, min_face_size = self.fix_param
        # load pnet model
        detectors = [None, None, None]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet
        self.aux_idx = 0
        # load rnet model
        if "onet" in test_mode or "rnet" in test_mode:
            if is_ERC:
                self.aux_idx = 4
                RNet = Detector(R_Net_ERC, 24, batch_size[1], model_path[1], self.aux_idx)
            else:
                RNet = Detector(M_Net, 18, batch_size[1], model_path[1])
                #RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
            detectors[1] = RNet

        # load onet model
        if "onet" in test_mode:
            if 'landmark_pose' in test_mode:
                self.aux_idx = 3
                ONet = Detector(O_AUX_Net, 48, batch_size[2], model_path[2], self.aux_idx)
            else:
                ONet = Detector(O_Net, 48, batch_size[2], model_path[2])

            detectors[2] = ONet

        jdap_detector = JDAPDetector(detectors=detectors, is_ERC=False, min_face_size=min_face_size, threshold=thresh)
        return jdap_detector

    def show_result(self, image, results):
        cal_boxes = results[0] if type(results) is tuple else results
        box_num = cal_boxes.shape[0]

        for i in range(box_num):
            # Face rect and score
            bbox = cal_boxes[i, :4]
            score = cal_boxes[i, 4]
            self.draw_rectangle(image, bbox)
            self.draw_text(image, bbox, score)

            # Face landmark
            if self.aux_idx == 3 or self.aux_idx == 1:
                land_point = np.round(results[1][i]).astype(dtype=np.int32)
                self.draw_point(image, land_point, 5)
                # point_x = int(land_reg[i][land_id * 2] * proposal_side + src_boxes[i][0])
                # point_y = int(land_reg[i][land_id * 2 + 1] * proposal_side + src_boxes[i][1])
                # for p in range(5):
                #     cv2.circle(image, (land_point[p], land_point[5+p]), 2, (200, 0, 0), 2)

            # Head pose
            if self.aux_idx == 3 or self.aux_idx == 2:
                head_pose = results[2][i] * 180 / 3.14
                self.draw_text(image, bbox, head_pose)

        cv2.imshow("show result", image)
        cv2.waitKey(0)

    def draw_rectangle(self, image, rect, color=(0, 0, 200)):
        rect = [int(x) for x in rect]
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

    def draw_point(self, image, points, point_num, color=(200, 0, 0)):
        for p in range(point_num):
            cv2.circle(image, (points[p], points[5 + p]), 2, color, 2)

    def draw_text(self, image, rect, text, color=(200, 0, 200)):
        rect = [int(x) for x in rect]
        if len(text.shape):
            s = ' '.join([str('%.2f' % x) for x in text])
            # Draw top left
            cv2.putText(image, s, (rect[0], rect[1]), 1, 1, color, 2)
        else:
            s = str('%.4f' % text)
            # Draw center
            cv2.putText(image, s, ((rect[0]+rect[2])//2, (rect[1]+rect[3])//2), 1, 1, color)

    def detect(self, image):
        return self.jdap_detector.detect(image, self.aux_idx)


def test_aux_net(name_list, dataset_path, prefix, epoch, batch_size, test_mode="rnet",
                 thresh=[0.6, 0.6, 0.7], min_face_size=24, vis=False):

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

    mtcnn_detector = JDAPDetector(detectors=detectors, min_face_size=min_face_size, threshold=thresh)

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
            write_str += ' '.join(str(x) for x in [x1, y1, x2 - x1 + 1, y2 - y1 + 1]) + ' %.4f' % score + '\n'
            if vis:
                pose_info = "%.2f %.2f %.2f" % (pose_reg[i][0], pose_reg[i][1], pose_reg[i][2])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 2)
                cv2.putText(image, pose_info, (x1, y2), 1, 1, (200, 200, 0), 2)
                for land_id in range(5):
                    point_x = int(land_reg[i][land_id * 2] * proposal_side[i] + src_boxes[i][0])
                    point_y = int(land_reg[i][land_id * 2 + 1] * proposal_side[i] + src_boxes[i][1])
                    # point_x = int(land_reg[i][land_id * 2] * w + x1)
                    # point_y = int(land_reg[i][land_id * 2 + 1] * h + y1)
                    cv2.circle(image, (point_x, point_y), 2, (200, 0, 0), 2)
        fout.write(write_str)
        if vis:
            cv2.imshow("a", image)
            cv2.waitKey(0)


def test_single_net(image_list, prefix, epoch, stage, attribute=False):
    model_path = '%s-%s' % (prefix, epoch)
    # load pnet model
    if stage == 12:
        detector = FcnDetector(P_Net, model_path)
    elif stage == 24:
        detector = Detector(R_Net, 24, 1, model_path)
    elif stage == 48:
        if attribute:
            detector = Detector(O_AUX_Net, 48, 1, model_path, aux_idx=3)
        else:
            detector = Detector(O_Net, 48, 1, model_path)
    for image_name in image_list:
        image = cv2.imread(image_name)
        norm_img = (image.copy() - 127.5) * 0.0078125  # C H W
        norm_img = norm_img[np.newaxis, :]
        fms = detector.predict(norm_img)
        # Print end_points keys
        print(fms[-1].keys())
        t = fms[-1]
        print(t)