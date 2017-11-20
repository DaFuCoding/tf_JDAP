import numpy as np
import tensorflow as tf


def cls_ERC(sess, cls_prob, cls_label_int64):
    """ Using ERC in face classification.
        positive face label is +1
        negative face label is 0
        part face label     is -1
    """
    #t = FLAGS.ERC_thresh
    t = 0.01
    label = tf.cast(cls_label_int64, tf.int64)
    valid_inds = tf.where(label >= 0)
    valid_cls_label = tf.gather(label, valid_inds)
    valid_cls_prob = tf.gather(cls_prob, valid_inds)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=valid_cls_prob, labels=valid_cls_label)
    # Reserve part samples
    easy_prob = tf.nn.softmax(logits=cls_prob)
    DR1_index = tf.where(tf.logical_or(easy_prob[:, 1] > t, tf.not_equal(label, 0)))
    loss = tf.reshape(loss, (-1,))
    ones = tf.ones_like(valid_cls_label, dtype=tf.float32)
    keep_num = tf.cast(tf.reduce_sum(ones), tf.int32)
    _, k_index = tf.nn.top_k(loss, keep_num)
    loss = tf.gather(loss, k_index)
    return tf.reduce_mean(loss)


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

# + + - - +
cls_prob = np.array([[0.3, 0.7], [0.002, 0.998], [0.999, 0.001], [0.99, 0.01]], dtype=np.float32)
# + + - + p
cls_label = np.array([1, 1, 0, 1], dtype=np.int64)
pos_index = tf.squeeze(tf.where(cls_label != 0), axis=1)
loss = cls_ERC(sess, cls_prob, cls_label)
#tensor = tf.convert_to_tensor(cls_prob, dtype=tf.float32)

#index = tf.gather(tensor, pos_index)
#print(sess.run(index))
