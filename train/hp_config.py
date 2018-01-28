import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("gpu_id", "0", "gpu device id.")
flags.DEFINE_string("logdir", "./logdir/pnet/pnet", "Some visual log information.")
flags.DEFINE_string("model_prefix", "./models/pnet/pnet_OHEM/pnet", "Checkpoint and model path")
flags.DEFINE_string("tfrecords_root", "/home/dafu/workspace/FaceDetect/tf_JDAP/tfrecords/pnet", "Train data.")
flags.DEFINE_integer("tfrecords_num", 4, "Sum of tfrecords.")
flags.DEFINE_integer("image_size", 12, "Netwrok input")
flags.DEFINE_integer("frequent", 100, "Show train result in frequent.")
flags.DEFINE_integer("image_sum", 1031327, "Sum of images in train set.")
flags.DEFINE_integer("val_image_sum", 271973, "Sum of images in val set.")
# Hyper parameter
flags.DEFINE_integer("batch_size", 128, "Batch size in classification task.")
flags.DEFINE_float("lr", 0.01, "Base learning rate.")
flags.DEFINE_float("lr_decay_factor", 0.1, "learning rate decay factor.")

flags.DEFINE_integer("end_epoch", 16, "How many epoch training.")

########################################
# Hard sample mining and Loss Function #
########################################
flags.DEFINE_boolean("is_ohem", True, "Whether using ohem in face classification.")
flags.DEFINE_float("ohem_ratio", 0.7, "")
flags.DEFINE_string("loss_type", "SF", "SF: SoftmaxWithLoss. FL: Focal loss.")
flags.DEFINE_float("fl_gamma", 2, "")
flags.DEFINE_boolean("fl_balance", False, "Whether using alpha balance positive and negative.")
flags.DEFINE_float("fl_alpha", 0.25, "")
flags.DEFINE_boolean("is_ERC", False, "Using early rejecting classifier.")
flags.DEFINE_float("ERC_thresh", 0.01, "ERC thresh.")
######################
# Optimization Flags #
######################
flags.DEFINE_string("optimizer", "adam", "Optimization strategy.")
flags.DEFINE_float("adam_beta1", 0.9, "")
flags.DEFINE_float("adam_beta2", 0.999, "")
flags.DEFINE_float("adam_epsilon", 1e-8, "")
flags.DEFINE_float("momentum", 0.9, "A `Tensor` or a floating point value.  The momentum.")

# Logging parameter
flags.DEFINE_boolean("is_feature_visual", True, "Output feature map per frequent")

# Attribute parameter
flags.DEFINE_integer("landmark_num", 68, "Landmark regression points")
