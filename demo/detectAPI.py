#_*_ coding: utf-8 _*_
import os

import numpy as np
import cv2

from nets.JDAP_Net import JDAP_12Net as P_Net
from nets.JDAP_Net import JDAP_24Net as R_Net
from nets.JDAP_Net import JDAP_24Net_ERC as R_Net_ERC
from nets.JDAP_Net import JDAP_48Net as O_Net
from nets.JDAP_Net import JDAP_48Net_Lanmark_Pose as O_AUX_Net
from fcn_detector import FcnDetector
from jdap_detect import JDAPDetector
from detector import Detector

__all__ = ['FDDB', 'WIDER', 'CelebA', 'AFLW', 'L300WP']


class DetectAPI(object):
    def __init__(self, image_name_file,
                 prefix, epoch, test_mode, batch_size, is_ERC, thresh, min_face_size):
        self.label_infos = self.get_labels(image_name_file)
        # To do compare experiments
        self.fix_param = [test_mode, batch_size, is_ERC, thresh, min_face_size]
        self.jdap_detector = self._init_model(prefix, epoch)

    def get_labels(self, image_name_file):
        if not os.path.exists(image_name_file):
            raise ValueError("File is not exist.")
        with open(image_name_file, 'r') as f:
            lines = f.readlines()
        return lines

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
                RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
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
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.putText(image, '%.2f' % score, ((x1 + x2) // 2, (y1 + y2) // 2), 1, 1, (0, 0, 200), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 2)

            # Face landmark
            if self.aux_idx == 3 or self.aux_idx == 1:
                land_point = np.round(results[1][i]).astype(dtype=np.int32)
                # point_x = int(land_reg[i][land_id * 2] * proposal_side + src_boxes[i][0])
                # point_y = int(land_reg[i][land_id * 2 + 1] * proposal_side + src_boxes[i][1])
                for p in range(5):
                    cv2.circle(image, (land_point[p], land_point[5+p]), 2, (200, 0, 0), 2)

            # Head pose
            if self.aux_idx == 3 or self.aux_idx == 2:
                head_pose = results[2][i] * 180 / 3.14
                pose_info = "%.2f %.2f %.2f" % (head_pose[0], head_pose[1], head_pose[2])
                cv2.putText(image, pose_info, (x1, y2), 1, 1, (200, 200, 0), 2)

        cv2.imshow("show result", image)
        cv2.waitKey(0)

    def get_rect_str_result(self, cal_boxes):
        rect_str_result = ''
        for bbox in range(cal_boxes):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            rect_str_result += ' '.join(str(x) for x in [x1, y1, x2 - x1 + 1, y2 - y1 + 1]) + ' %.4f' % bbox[4] + '\n'
        return rect_str_result

    def label_parser(self, label_info):
        raise NotImplementedError()

    def detect(self, image):
        return self.jdap_detector.detect(image, self.aux_idx)

    def do_eval(self, output_file):
        """ Ouput detect result in different regular.
        """
        pass


class FDDB(DetectAPI):
    def __init__(self, dataset_path, vis,
                 image_name_file, prefix, epoch, batch_size, test_mode="rnet",
                 is_ERC=False, thresh=[0.6, 0.6, 0.7], min_face_size=24):
        super(FDDB, self).__init__(image_name_file, prefix, epoch, batch_size, test_mode,
                                   is_ERC, thresh, min_face_size)
        self.dataset_path = dataset_path
        self.vis = vis

    # TODO: parser regular
    def label_parser(self, label_info):
        info = label_info.strip().split()
        image_name = info[0]
        if '.jpg' not in image_name:
            image_name += '.jpg'
        return os.path.join(self.dataset_path, image_name)

    def get_image_name(self, label_info):
        image_name = label_info.strip().split()[0]
        if '.jpg' not in image_name:
            image_name += '.jpg'
        return os.path.join(self.dataset_path, image_name)

    def do_eval(self, output_file):
        with open(output_file, 'w') as fout:
            for label_info in self.label_infos:
                image_name = self.get_image_name(label_info)
                print(image_name)
                image = cv2.imread(image_name)
                # All output result
                results = self.detect(image)
                cal_boxes = results[0] if type(results) is tuple else results

                box_num = cal_boxes.shape[0]
                x = cal_boxes[:, 0]
                y = cal_boxes[:, 1]
                w = cal_boxes[:, 2] - cal_boxes[:, 0] + 1
                h = cal_boxes[:, 3] - cal_boxes[:, 1] + 1
                scores = cal_boxes[:, 4]
                # label_info including \n
                write_result = '%s%d\n' % (label_info, box_num)
                for i in range(box_num):
                    # Face rect and score
                    write_result += '%d %d %d %d %.4f\n' % (x[i], y[i], w[i], h[i], scores[i])
                fout.write(write_result)



class WIDER(DetectAPI):
    pass


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