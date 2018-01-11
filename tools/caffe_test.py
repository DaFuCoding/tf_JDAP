import sys
caffe_root = '/home/dafu/workspace/Compression/caffe_ristretto/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
import numpy as np
import glob
from collections import OrderedDict
from utils import *


def consistency(net_size, deploy, caffemodel, img_list):
    """
    Consistency test in tf2caffe model
    Image size is net_size X net_size X 3
    Returns:
        Feature map in different layer
    """
    net = caffe.Net(deploy, caffemodel, caffe.TEST)
    caffe.set_mode_cpu()
    feature_maps = list()
    for img_path in img_list:
        fm = OrderedDict()
        img = cv2.imread(img_path)
        caffe_img = (img.copy() - 127.5) * 0.0078125  # H W C
        scale_img = np.transpose(caffe_img, [2, 0, 1])  # C H W
        net.blobs['data'].reshape(1, 3, net_size, net_size)
        net.blobs['data'].data[...] = scale_img
        net.forward()
        for layer_name in net.blobs.keys():
            fm[layer_name] = net.blobs[layer_name].data.copy()
        feature_maps.append(fm)
    return feature_maps


def _candidate_arrange(dets, img, height, width, net_size):

    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, width, height)
    num_boxes = dets.shape[0]

    cropped_ims = np.zeros((num_boxes, 3, net_size, net_size), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        norm_img = resize_image_by_wh(tmp, (net_size, net_size))
        norm_img = np.transpose(norm_img, [2, 0, 1])  # C H W
        cropped_ims[i, :, :, :] = norm_img
    return cropped_ims


def inference(deploys, caffemodels, image_list, thresh):
    nets = list()
    for deploy, caffemodel in zip(deploys, caffemodels):
        nets.append(caffe.Net(deploy, caffemodel, caffe.TEST))
    caffe.set_mode_cpu()
    for img_path in image_list:
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        scales, scales_wh = calc_scale(height, width, min_face_size=24)
        all_boxes = list()

        # PNet
        for scale, tuple_wh in zip(scales, scales_wh):
            resized_img = resize_image_by_wh(img, tuple_wh)
            resized_img = np.transpose(resized_img, [2, 0, 1])  # C H W
            nets[0].blobs['data'].reshape(1, 3, tuple_wh[1], tuple_wh[0])
            nets[0].blobs['data'].data[...] = resized_img
            output = nets[0].forward()
            cls_prob = output['prob1'][0][1]
            bbox_reg = output['conv4-2'][0]
            bbox_reg = np.transpose(bbox_reg, [1, 2, 0])
            boxes = generate_bbox(cls_prob, bbox_reg, scale, thresh[0])
            # Numpy slice without security check
            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]  # if keep is [], boxes is also []
            if boxes.size == 0:
                continue
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

        # RNet
        num_boxes = len(boxes_c)
        if num_boxes == 0:
            return None, None
        net_size = 24
        nets[1].blobs['data'].reshape(num_boxes, 3, net_size, net_size)
        dets = convert_to_square(boxes_c)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        cropped_ims = _candidate_arrange(dets, img, height, width, net_size)

        nets[1].blobs['data'].data[...] = cropped_ims
        result = nets[1].forward()
        keep_inds = np.where(result['prob1'][:, 1] > thresh[1])

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = result['prob1'][:, 1][keep_inds]
            bbox_reg = result['conv5-2'][keep_inds]

        keep = py_nms(boxes, 0.7)
        boxes = boxes[keep]
        boxes_c = calibrate_box(boxes, bbox_reg[keep])

        # ONet
        num_boxes = len(boxes_c)
        if num_boxes == 0:
            return None, None
        net_size = 48
        nets[2].blobs['data'].reshape(num_boxes, 3, net_size, net_size)
        dets = convert_to_square(boxes_c)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        cropped_ims = _candidate_arrange(dets, img, height, width, net_size)
        nets[2].blobs['data'].data[...] = cropped_ims
        result = nets[2].forward()
        keep_inds = np.where(result['prob1'][:, 1] > thresh[2])

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = result['prob1'][:, 1][keep_inds]
            bbox_reg = result['conv6-2'][keep_inds]
            landmark = result['conv6-3'][keep_inds]
            pose_reg = result['conv6-4'][keep_inds]

        keep = py_nms(boxes, 0.7, 'Minimum')
        boxes = boxes[keep]
        landmark = landmark[keep]
        pose_reg = pose_reg[keep]
        boxes_c = calibrate_box(boxes, bbox_reg[keep])
        landmark = np.reshape(landmark, [-1, 5, 2])

        for i, box_c in enumerate(boxes_c):
            # Ouput Landmark
            box = boxes[i]
            cand_w = box[2] - box[0]
            cand_h = box[3] - box[1]
            landmark_scale_w = landmark[i][:, 0]
            landmark_scale_h = landmark[i][:, 1]
            points_2d_x = np.round(box[0] + cand_w * landmark_scale_w)
            points_2d_y = np.round(box[1] + cand_h * landmark_scale_h)
            for point_id in range(5):
                cv2.circle(img, (points_2d_x[point_id], points_2d_y[point_id]), 1, (200, 0, 0), 2)
            # Ouput Head Pose
            pose_angle = pose_reg[i] * 180 / 3.14
            pitch, yaw, roll = pose_angle
            pose_str = '%.2f %.2f %.2f' % (pitch, yaw, roll)

            x1, y1, x2, y2 = [int(x) for x in box_c[:4]]
            # Show information
            cv2.putText(img, pose_str, (x1, y1), 1, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 0), 2)
        cv2.imshow('a', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    deploys = ['/home/dafu/workspace/FaceDetect/tf_JDAP/models/MTCNN_Official/det1.prototxt',
              '/home/dafu/workspace/FaceDetect/tf_JDAP/models/MTCNN_Official/det2.prototxt',
               '/home/dafu/workspace/FaceDetect/tf_JDAP/models/MTCNN_Official/det3_landmark_pose.prototxt']

    caffemodels = ['tf2caffe_pnet.caffemodel', 'tf2caffe_rnet.caffemodel', 'tf2caffe_onet_landmark_pose.caffemodel']
    stage = 24
    # img_list = glob.glob('/home/dafu/Pictures/test/%d_*.jpg' % stage)
    # caffe_fms = consistency(stage, deploys[-1], caffemodels[-1], img_list)
    # print(caffe_fms[0]['conv4'])
    # print(caffe_fms)
    image_list = glob.glob('/home/dafu/data/FDDB/2002/07/19/big/*.jpg')
    inference(deploys, caffemodels, image_list, [0.6, 0.5, 0.5])
