import tensorflow as tf

BATCH_NUM = 1
CHANNEL = 3


class FcnDetector(object):
    def __init__(self, net_factory, model_path):
        with tf.Graph().as_default():
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            if BATCH_NUM is not 1 or CHANNEL is not 3:
                raise Exception("PNet using FCN, it only support 1 batch size. Channel only support 3.")
            image_reshape = tf.reshape(self.image_op, [BATCH_NUM, self.height_op, self.width_op, CHANNEL])
            self.cls_prob, self.bbox_pred, self.end_points = net_factory(image_reshape, is_training=False, mode='TEST')
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)

    def predict(self, image):
        height, width, _ = image.shape
        cls_prob, bbox_pred, end_points = self.sess.run([self.cls_prob, self.bbox_pred, self.end_points],
                                                        feed_dict={self.image_op: image, self.width_op: width,
                                                                   self.height_op: height})
        return cls_prob, bbox_pred, end_points