import cv2
import time
import numpy as np
from configs.config import config
from tools.utils import py_nms, generate_bbox, resize_image_by_wh, calc_scale
import matplotlib.pyplot as plt
import random


class JDAPDetector(object):
    def __init__(self,
                 detectors,
                 is_ERC=False,
                 min_face_size=24,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 slide_window=False):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.is_ERC = is_ERC
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window
        self.p_end_points = []
        self.r_end_points = []

    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h,w)
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1,  bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx , dy= np.zeros((num_box, )), np.zeros((num_box, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

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
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        height, width, channel = image.shape
        scales, scales_wh = calc_scale(height, width, self.min_face_size)
        self.p_end_points = []
        # FCN in PNet
        all_boxes = list()
        for current_scale, tuple_wh in zip(scales, scales_wh):
            im_resized = resize_image_by_wh(image, tuple_wh)
            cls_map, reg, end_points = self.pnet_detector.predict(im_resized)
            self.p_end_points.append(end_points)
            boxes = generate_bbox(cls_map[0, :, :, 1], reg, current_scale, self.thresh[0])
            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c

    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting RNet batch size
        batch_size = self.rnet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset RNet batch size if this info appears frequently, \
face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) * 0.0078125
        if self.is_ERC:
            cls_scores, reg, reserve_mask = self.rnet_detector.predict(cropped_ims)
        else:
            cls_scores, reg = self.rnet_detector.predict(cropped_ims)
        keep_inds = np.where(cls_scores[:, 1] > self.thresh[1])

        if len(keep_inds) > 0:
            if self.is_ERC:
                boxes = dets[reserve_mask][keep_inds]
            else:
                boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[:, 1][keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = py_nms(boxes, 0.7)
        boxes = boxes[keep]

        boxes_c = self.calibrate_box(boxes, reg[keep])

        return boxes, boxes_c

    def detect_onet(self, im, dets, aux_idx):
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
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting ONet batch size
        batch_size = self.onet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset ONet batch size if this info appears frequently, \
face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = im[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) * 0.0078125
        if aux_idx == 0:
            cls_scores, reg = self.onet_detector.predict(cropped_ims)
        elif aux_idx == 1:
            pass
        elif aux_idx == 2:
            pass
        elif aux_idx == 3:
            cls_scores, reg, land_reg, pose_reg = self.onet_detector.predict(cropped_ims)
        keep_inds = np.where(cls_scores[:, 1] > self.thresh[2])

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[:, 1][keep_inds]
            reg = reg[keep_inds]
        else:
            if aux_idx == 3:
                return None, None, None, None
            return None, None

        boxes_c = self.calibrate_box(boxes, reg)

        keep = py_nms(boxes_c, 0.7, "Minimum")
        boxes_c = boxes_c[keep]
        if aux_idx == 3:
            land_reg = land_reg[keep]
            pose_reg = pose_reg[keep]
            return boxes[keep], boxes_c, land_reg, pose_reg
        return boxes, boxes_c

    def detect(self, img, aux_idx=0):
        """Detect face over image
        """
        boxes = None
        t = time.time()

        # pnet
        if self.pnet_detector:
            boxes, boxes_c = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes, boxes_c = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            if aux_idx == 0:
                boxes, boxes_c = self.detect_onet(img, boxes_c, aux_idx)
            if boxes_c is None:
                return np.array([])
            elif aux_idx == 3:
                boxes, boxes_c, land_reg, pose_reg = self.detect_onet(img, boxes_c, aux_idx)
                if boxes_c is None:
                    return np.array([]), np.array([]), np.array([]), np.array([])
                else:
                    return boxes, boxes_c, land_reg, pose_reg

            t3 = time.time() - t
            t = time.time()
            print "time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3)

        return boxes_c
#
#     def detect_face(self, imdb, test_data, vis):
#         """Detect face over image
#
#         Parameters:
#         ----------
#         imdb: imdb
#             image database
#         test_data: data iter
#             test data iterator
#         vis: bool
#             whether to visualize detection results
#
#         Returns:
#         -------
#         """
#         all_boxes = list()
#         batch_idx = 0
#         for databatch in test_data:
#             if batch_idx % 100 == 0:
#                 print "%d images done" % batch_idx
#             im = databatch
#             t = time.time()
#
#             # pnet
#             if self.pnet_detector:
#                 boxes, boxes_c = self.detect_pnet(im)
#                 if boxes_c is None:
#                     all_boxes.append(np.array([]))
#                     batch_idx += 1
#                     continue
#                 if vis:
#                     rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#                     self.vis_two(rgb_im, boxes, boxes_c)
#
#                 t1 = time.time() - t
#                 t = time.time()
#
#             # rnet
#             if self.rnet_detector:
#                 boxes, boxes_c = self.detect_rnet(im, boxes_c)
#                 if boxes_c is None:
#                     all_boxes.append(np.array([]))
#                     batch_idx += 1
#                     continue
#                 if vis:
#                     self.vis_two(rgb_im, boxes, boxes_c)
#
#                 t2 = time.time() - t
#                 t = time.time()
#
#             # onet
#             if self.onet_detector:
#                 boxes, boxes_c = self.detect_onet(im, boxes_c)
#                 if boxes_c is None:
#                     all_boxes.append(np.array([]))
#                     batch_idx += 1
#                     continue
# #                all_boxes.append(boxes_c)
#                 if vis:
#                     self.vis_two(rgb_im, boxes, boxes_c)
#
#                 t3 = time.time() - t
#                 t = time.time()
#                 print "time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3)
#
#             all_boxes.append(boxes_c)
#             batch_idx += 1
#             # if batch_idx == 2:
#             #     imdb.write_results(all_boxes)
#             #     return all_boxes
#         # save detections into fddb format
#         imdb.write_results(all_boxes)
#         return all_boxes