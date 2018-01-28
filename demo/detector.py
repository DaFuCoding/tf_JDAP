import tensorflow as tf
import numpy as np
from collections import OrderedDict


class Detector(object):
    def __init__(self, net_factory, data_size, batch_size, model_path, aux_idx=0):
        self._aux_idx = aux_idx
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[None, data_size, data_size, 3], name='input_image')
            if self._aux_idx == 0:
                self.cls_prob, self.bbox_pred, self.end_points = net_factory(self.image_op, is_training=False)
            # Only face landmark aux
            if self._aux_idx == 1:
                self.cls_prob, self.bbox_pred, self.land_pred = net_factory(self.image_op, is_training=False)
            # Only head pose aux
            elif self._aux_idx == 2:
                self.cls_prob, self.bbox_pred, self.pose_pred = net_factory(self.image_op, is_training=False)
            # face landmark and head pose aux together
            elif self._aux_idx == 3:
                self.cls_prob, self.bbox_pred, self.pose_pred, self.land_pred, self.end_points\
                    = net_factory(self.image_op, is_training=False)
            # Using early reject classifier
            elif self._aux_idx == 4:
                self.cls_prob, self.bbox_pred, self.DR1_index, self.DR2_index = net_factory(self.image_op, is_training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size

    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        batch_size = self.batch_size

        minibatch = []
        cur = 0
        n = databatch.shape[0]
        while cur < n:
            minibatch.append(databatch[cur:min(cur+batch_size, n), :, :, :])
            cur += batch_size
        cls_prob_list = []
        bbox_pred_list = []
        land_pred_list = []
        pose_pred_list = []
        #end_points_dict = OrderedDict()
        DR_index_list = []
        if self._aux_idx == 4:
            cls_prob, bbox_pred, DR1_index, DR2_index = \
                self.sess.run([self.cls_prob, self.bbox_pred, self.DR1_index, self.DR2_index],
                              feed_dict={self.image_op: databatch})
            last_index = DR1_index[DR2_index]
            return cls_prob, bbox_pred, last_index
        else:
            for idx, data in enumerate(minibatch):
                m = data.shape[0]
                real_size = self.batch_size
                if m < batch_size:
                    keep_inds = np.arange(m)
                    gap = self.batch_size - m
                    while gap >= len(keep_inds):
                        gap -= len(keep_inds)
                        keep_inds = np.concatenate((keep_inds, keep_inds))
                    if gap != 0:
                        keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                    data = data[keep_inds]
                    real_size = m
                if self._aux_idx == 0:
                    cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred], feed_dict={self.image_op: data})
                elif self._aux_idx == 1:
                    cls_prob, bbox_pred, land_pred = \
                        self.sess.run([self.cls_prob, self.bbox_pred, self.land_pred], feed_dict={self.image_op: data})
                    land_pred_list.append(land_pred[:real_size])
                elif self._aux_idx == 2:
                    cls_prob, bbox_pred, pose_pred = \
                        self.sess.run([self.cls_prob, self.bbox_pred, self.pose_pred], feed_dict={self.image_op: data})
                    pose_pred_list.append(pose_pred[:real_size])
                elif self._aux_idx == 3:
                    cls_prob, bbox_pred, pose_pred, land_pred = \
                        self.sess.run([self.cls_prob, self.bbox_pred, self.pose_pred, self.land_pred],
                                      feed_dict={self.image_op: data})
                    land_pred_list.append(land_pred[:real_size])
                    pose_pred_list.append(pose_pred[:real_size])
                # elif self._aux_idx == 4:
                #     cls_prob, bbox_pred, DR1_index, DR2_index = \
                #         self.sess.run([self.cls_prob, self.bbox_pred, self.DR1_index, self.DR2_index], feed_dict={self.image_op: data})
                #     index_1 = np.where(DR1_index < real_size)
                #     filter_num_1 = np.size(index_1, axis=1)
                #     #DR1_index_list.append(DR1_index[:filter_num_1])
                #     valid_base_index = DR1_index[:filter_num_1]
                #     index_2 = np.where(DR2_index < filter_num_1)
                #     filter_num_2 = np.size(index_2, axis=1)
                #     #DR2_index_list.append(DR2_index[:filter_num_2])
                #     short_index = DR2_index[:filter_num_2]
                #     last_index = valid_base_index[short_index]
                #     mask = np.zeros(batch_size, dtype=np.int32)
                #     mask[last_index] = 1
                #     cls_prob_list.append(cls_prob[short_index])
                #     bbox_pred_list.append(bbox_pred[short_index])
                #     DR_index_list.append(mask[:real_size])

                cls_prob_list.append(cls_prob[:real_size])
                bbox_pred_list.append(bbox_pred[:real_size])
                # for k, v in end_points.items():
                #     if k not in end_points:
                #         end_points[k] = v
                #     else:
                #         end_points[k] = np.concatenate((end_points[k], v), axis=0)
            if len(cls_prob_list):
                cls_result = np.concatenate(cls_prob_list, axis=0)
                bbox_result = np.concatenate(bbox_pred_list, axis=0)
            else:
                cls_result = []
                bbox_result = []

        if self._aux_idx == 0:
            return cls_result, bbox_result
        elif self._aux_idx == 1:
            land_result = []
            if len(land_pred_list):
                land_result = np.concatenate(land_pred_list, axis=0)
            return cls_result, bbox_result, land_result
        elif self._aux_idx == 2:
            pose_result = []
            if len(pose_pred_list):
                pose_result = np.concatenate(pose_pred_list, axis=0)
            return cls_result, bbox_result, pose_result
        elif self._aux_idx == 3:
            land_result = []
            pose_result = []
            if len(land_pred_list):
                land_result = np.concatenate(land_pred_list, axis=0)
                pose_result = np.concatenate(pose_pred_list, axis=0)
            return cls_result, bbox_result, pose_result, land_result
        # elif self._aux_idx == 4:
        #     # DR1_index_list = np.concatenate(DR1_index_list, axis=0)
        #     # DR2_index_list = np.concatenate(DR2_index_list, axis=0)
        #     # return cls_result, bbox_result, DR1_index_list, DR2_index_list
        #     reserve_mask = np.concatenate(DR_index_list, axis=0)
        #     return cls_result, bbox_result, reserve_mask