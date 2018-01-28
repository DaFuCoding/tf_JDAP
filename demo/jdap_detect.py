import time
from tools.utils import *
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class JDAPDetector(object):
    def __init__(self,
                 detectors,
                 is_ERC=False,
                 min_face_size=24,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.is_ERC = is_ERC
        self.min_face_size = min_face_size
        self.thresh = threshold
        self.scale_factor = scale_factor

    def _candidate_arrange(self, dets, img, net_size):
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, self.width, self.height)
        num_boxes = dets.shape[0]

        cropped_ims = np.zeros((num_boxes, net_size, net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = resize_image_by_wh(tmp, (net_size, net_size))
        return cropped_ims

    def detect_pnet(self, image):
        """Get face candidates through pnet

        Parameters:
        ----------
        image: numpy array
            input image array

        Intermediate:
        ----------
        end_points: numpy array
            feature map differ layer

        Returns:
        -------
        ## Detracted boxes
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        # FCN in PNet
        all_boxes = list()
        for current_scale, tuple_wh in zip(self.scales, self.scales_wh):
            im_resized = resize_image_by_wh(image, tuple_wh)
            cls_map, bbox_reg, end_points = self.pnet_detector.predict(im_resized)
            boxes = generate_bbox(cls_map[0, :, :, 1], bbox_reg[0], current_scale, self.thresh[0])
            # Numpy slice without security check
            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]  # if keep is [], boxes is also []
            if boxes.size == 0:
                continue
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        #boxes = all_boxes[:, :5]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        return boxes_c
        #return boxes, boxes_c

    def detect_rnet(self, image, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        ## Detracted boxes
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        cropped_ims = self._candidate_arrange(dets, image, self.rnet_detector.data_size)

        if self.is_ERC:
            cls_scores, reg, reserve_mask = self.rnet_detector.predict(cropped_ims)
        else:
            cls_scores, reg = self.rnet_detector.predict(cropped_ims)
        if len(cls_scores) == 0:
            return None
        keep_inds = np.where(cls_scores[:, 1] > self.thresh[1])
        if len(keep_inds) == 0:
            return None

        if self.is_ERC:
            boxes = dets[reserve_mask][keep_inds]
        else:
            boxes = dets[keep_inds]

        boxes[:, 4] = cls_scores[:, 1][keep_inds]
        reg = reg[keep_inds]
        keep = py_nms(boxes, 0.7)
        boxes = boxes[keep]

        boxes_c = calibrate_box(boxes, reg[keep])

        return boxes_c

    def detect_onet(self, image, dets, aux_idx):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        cropped_ims = self._candidate_arrange(dets, image, self.onet_detector.data_size)

        if aux_idx == 0:
            cls_scores, bbox_reg = self.onet_detector.predict(cropped_ims)
        elif aux_idx == 1:  # Landmark
            cls_scores, bbox_reg, land_reg = self.onet_detector.predict(cropped_ims)
        elif aux_idx == 2:  # Head pose
            cls_scores, bbox_reg, pose_reg = self.onet_detector.predict(cropped_ims)
        elif aux_idx == 3:  # Landmark and Head pose
            cls_scores, bbox_reg, pose_reg, land_reg = self.onet_detector.predict(cropped_ims)
        if len(cls_scores) == 0:
            return None, None, None
        keep_inds = np.where(cls_scores[:, 1] > self.thresh[2])
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[:, 1][keep_inds]
            bbox_reg = bbox_reg[keep_inds]
            if aux_idx == 3:
                land_reg = land_reg[keep_inds]
                pose_reg = pose_reg[keep_inds]
            elif aux_idx == 1:
                land_reg = land_reg[keep_inds]
            elif aux_idx == 2:
                pose_reg = pose_reg[keep_inds]
        else:
            if aux_idx == 3:
                return None, None, None
            elif aux_idx != 0:
                return None, None
            return None

        boxes_c = calibrate_box(boxes, bbox_reg)

        keep = py_nms(boxes_c, 0.7, "Minimum")
        boxes_c = boxes_c[keep]

        boxes = boxes[keep]
        side_w = boxes[:, 2] - boxes[:, 0] + 1
        side_h = boxes[:, 3] - boxes[:, 1] + 1
        if aux_idx == 3:
            land_reg = land_reg[keep]
            land_reg = np.reshape(land_reg, [-1, FLAGS.landmark_num, 2])
            # Point format transfer to [x x x x x y y y y y]
            land_point = np.transpose(
                np.vstack((boxes[:, 0] + land_reg[:, :, 0].T * side_w,
                           boxes[:, 1] + land_reg[:, :, 1].T * side_h))
            )
            pose_reg = pose_reg[keep]
            return boxes_c, pose_reg, land_point
        elif aux_idx == 1:
            land_reg = land_reg[keep]
            land_reg = np.reshape(land_reg, [-1, FLAGS.landmark_num, 2])
            # Point format transfer to [x x x x x y y y y y]
            land_point = np.transpose(
                np.vstack((boxes[:, 0] + land_reg[:, :, 0].T * side_w,
                           boxes[:, 1] + land_reg[:, :, 1].T * side_h))
            )
            return boxes_c, land_point
        elif aux_idx == 2:
            pose_reg = pose_reg[keep]
            return boxes_c, pose_reg
        return boxes_c

    def detect(self, img, aux_idx=0):
        """Detect face in three stage
        """
        t = time.time()
        # Preprocessing
        self.height, self.width, self.channel = img.shape
        self.scales, self.scales_wh = calc_scale(self.height, self.width, self.min_face_size)
        # pnet
        if self.pnet_detector:
            boxes_c = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes_c = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            if boxes_c is None:
                return np.array([])

            if aux_idx == 0:
                boxes_c = self.detect_onet(img, boxes_c, aux_idx)
            elif aux_idx == 3:
                if boxes_c is None:
                    return np.array([]), np.array([]), np.array([])
                boxes_c, head_pose, land_point = self.detect_onet(img, boxes_c, aux_idx)
                return boxes_c, head_pose, land_point
            elif aux_idx == 2:
                if boxes_c is None:
                    return np.array([]), np.array([])
                boxes_c, head_pose = self.detect_onet(img, boxes_c, aux_idx)
                return boxes_c, head_pose
            elif aux_idx == 1:
                if boxes_c is None:
                    return np.array([]), np.array([])
                boxes_c, land_point = self.detect_onet(img, boxes_c, aux_idx)
                return boxes_c, land_point
            else:
                raise NotImplementedError("Not support aux_idx.")

        return boxes_c
        # Consume time
        # t3 = time.time() - t
        # print("time cost " + '{:.3f}s'.format(t1+t2+t3))
        # print('pnet {:.3f}s  rnet {:.3f}s  onet {:.3f}s'.format(t1, t2, t3))
