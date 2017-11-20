import numpy as np
import tensorflow as tf


inputs = tf.Variable(tf.zeros([7], dtype=tf.int32), trainable=False)
index1 = tf.convert_to_tensor([1, 2, 4, 6], dtype=tf.int32)
index2 = tf.convert_to_tensor([0, 3], dtype=tf.int32)

mask = tf.gather(index1, index2)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
init = tf.initialize_all_variables()
sess.run(init)
inputs = tf.scatter_update(inputs, mask, tf.ones_like(mask, dtype=tf.int32))
#inputs[mask] = 1
#print(sess.run(mask))
print(sess.run(inputs))

#print(sess.run(inputs))
#inputs[index1[index2]] = 1



