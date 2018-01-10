import sys
caffe_root = '/home/dafu/workspace/Compression/caffe_ristretto/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
import numpy as np
import glob
from collections import OrderedDict
from utils import calc_scale, resize_image_by_wh, py_nms

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

def inference(img_path):
    net = caffe.Net(deploy, caffemodel, caffe.TEST)
    caffe.set_mode_cpu()
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    scales, scales_wh = calc_scale(height, width, min_face_size=24)
    for scale, tuple_wh in zip(scales, scales_wh):
        resized_img = resize_image_by_wh(img, tuple_wh)
        resized_img = np.transpose(resized_img, [2, 0, 1])  # C H W
        net.blobs['data'].reshape(1, 3, tuple_wh[1], tuple_wh[0])
        net.blobs['data'].data[...] = resized_img
        output = net.forward()
        cls_prob = output['prob1'][0][1]
        boxes_reg = output['conv4-2'][0]

        # for box in boxes:
        #     x1, y1, x2, y2 = [int(x) for x in box[:4]]
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 0), 2)
        # cv2.imshow('a', img)
        # cv2.waitKey(0)


if __name__ == '__main__':
    deploy = '/home/dafu/workspace/FaceDetect/tf_JDAP/models/MTCNN_Official/det1.prototxt'
    caffemodel = 'tf2caffe_pnet.caffemodel'
    stage = 12
    img_list = glob.glob('/home/dafu/Pictures/test/%d_*.jpg' % stage)
    #caffe_fms = consistency(stage, deploy, caffemodel, img_list)
    #print(caffe_fms)
    inference('/home/dafu/Pictures/test/test.jpg')
"""

image_num = len(scales)
rectangles = []
for i in range(image_num):
    cls_prob = out[i]['prob1'][0][1]
    roi      = out[i]['conv4-2'][0]
    out_h, out_w = cls_prob.shape
    out_side = max(out_h, out_w)
    print i
    rectangle = detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
    if len(rectangle) != 0:
        rectangles.extend(rectangle)
rectangles = np.array(rectangles)
keep = py_nms(rectangles[:, :5], 0.7, 'Union')
boxes = rectangles[keep]

"""