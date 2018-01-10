"""
Brief:
    Trained-model in TF convert to Caffe model.
    Including 12, 24, 48(Net)
Future:
    MMdnn(https://github.com/Microsoft/MMdnn) replace

"""
import sys
caffe_root = '/home/dafu/workspace/Compression/caffe_ristretto/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os
from collections import OrderedDict


class TensorflowParser(object):
    """
    Brief:
        Now only support TF model convert to Caffe model in 'JDAP' algorithm.
        This isn't common model convert method.
    Network Op includes convolution and prelu and FC
    """

    def __init__(self, stage, tf_model_prefix, caffe_model_prefix):

        self._tf_model_prefix = tf_model_prefix
        self._caffe_model_prefix = caffe_model_prefix
        self._dest_nodes = None  # Ex: 'JDAP_12Net/conv1/weights'
        #self._tf_model = self.load_meta()
        self._ckpt_data = self.load_weights()
        # Keep valid weights in different stage
        self._stage = stage
        self.data = list()
        self.extract_valiad_weights()

    def load_protobuf_from_file(self, container, filename):
        with open(filename, 'rb') as fin:
            file_content = fin.read()

        # First try to read it as a binary file.
        try:
            container.ParseFromString(file_content)
            print("Parse file [%s] with binary format successfully." % (filename))
            return container

        except Exception as e:  # pylint: disable=broad-except
            print (
            "Info: Trying to parse file [%s] with binary format but failed with error [%s]." % (filename, str(e)))

        # Next try to read it as a text file.
        try:
            from google.protobuf import text_format
            text_format.Parse(file_content.decode('UTF-8'), container, allow_unknown_extension=True)
            print("Parse file [%s] with text format successfully." % (filename))
        except text_format.ParseError as e:
            raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

        return container

    def load_meta(self):
        """ Load a tensorflow meta file from disk

        Returns:
            model: A tensorflow protobuf file
        """

        from tensorflow.core.protobuf import meta_graph_pb2

        meta_graph = meta_graph_pb2.MetaGraphDef()
        self.load_protobuf_from_file(meta_graph, self._tf_model_prefix + '.meta')
        graph = meta_graph.graph_def
        if self._dest_nodes != None:
            from tensorflow.python.framework.graph_util import extract_sub_graph
            graph = extract_sub_graph(graph, self._dest_nodes.split(','))
        print ("Tensorflow model file [%s] loaded successfully." % self._tf_model_prefix)
        return graph

    def load_weights(self):
        """ Load a tensorflow checkpoint file from disk

        Returns:
            model: tensor name --> ndarry
        """

        reader = pywrap_tensorflow.NewCheckpointReader(self._tf_model_prefix)
        var_to_shape_map = reader.get_variable_to_shape_map()
        data = dict()
        for name in var_to_shape_map:
            tensor = reader.get_tensor(name)
            data[name] = tensor

        print ("Tensorflow checkpoint file [%s] loaded successfully. [%d] variables loaded."
               % (self._tf_model_prefix, len(data)))
        return data

    def map_valid_param_name(self):
        param_name = list()
        conv_num = 3
        conv_branch = []
        fc_num = 0
        attribute_name = []
        network_name = 'JDAP_%dNet' % self._stage
        prelu_name = 'prelu/alpha'
        conv_name = 'conv'
        fc_name = 'fc'
        conv_w_name = 'weights'
        conv_b_name = 'biases'

        if self._stage == 12:
            # express conv4_1 and conv4_2
            conv_branch = [[4, 2]]
        elif self._stage == 24:
            conv_num = 3
            fc_num = 3
        elif self._stage == 48:
            conv_num = 4
            fc_num = 3
            #attribute_name = ['landmark', 'pose_reg']
        else:
            print("Unknown stage.")
            return None

        def join_param(prefix_name):
            param_name.append(os.path.join(prefix_name, conv_w_name))
            param_name.append(os.path.join(prefix_name, conv_b_name))

        # Parser conv
        for id in range(conv_num):
            conv_pre = os.path.join(network_name, conv_name+str(id+1))
            join_param(conv_pre)
            param_name.append(os.path.join(conv_pre, prelu_name))
        # Parser conv-branch
        for branch_id in range(len(conv_branch)):
            branch_tuple = conv_branch[branch_id]
            for id in range(branch_tuple[1]):
                conv_pre = os.path.join(network_name, conv_name+'%d_%d' % (branch_tuple[0], id+1))
                join_param(conv_pre)
        # Parser FC
        for id in range(fc_num):
            fc_pre = os.path.join(network_name, fc_name + str(id+1))
            join_param(fc_pre)
            if id == 0:
                param_name.append(os.path.join(fc_pre, prelu_name))

        self.param_name = param_name

    def extract_valiad_weights(self):
        self.map_valid_param_name()
        for name in self.param_name:
            self.data.append((name, self._ckpt_data[name]))


class CaffeParser(object):
    # model = caffe_pb2.NetParameter()
    # f = open(caffe_model_name, 'rb')
    # model.ParseFromString(f.read())
    # f.close()
    # layers = model.layer
    # for layer in layers:
    #     print layer
    def __init__(self, net_name, model_name, tf_data, new_model_dir):
        #caffe.set_mode_cpu()
        self._net = caffe.Net(net_name, model_name, caffe.TEST)
        self.param_names = self._net.params.keys()
        self.tf_data = tf_data
        self.new_model_dir = new_model_dir

    def tf2caffe(self):
        param_cnt = 0
        for caffe_name in self.param_names:
            param = self._net.params[caffe_name]
            param_num = len(param)
            for i in range(param_num):
                print(param[i].data.shape)
                caffe_layer_data = param[i].data
                tf_layer_data = self.tf_data[param_cnt][1]
                # Caffe(Out In Kh Kw) TF(Kh Kw In Out) Reshape Caffe format
                if len(tf_layer_data.shape) == 4:
                    tf_layer_data = np.transpose(tf_layer_data, (3, 2, 0, 1))
                print(tf_layer_data.shape)
                assert (tf_layer_data.shape == caffe_layer_data.shape)
                param_cnt += 1
                # TF model assignment Caffe model
                self._net.params[caffe_name][i].data[:] = tf_layer_data
        self._net.save(self.new_model_dir)


tf_model_name = '/home/dafu/workspace/DLFramework/MMdnn/models/pnet-16'
caffe_model_name = '/home/dafu/workspace/FaceDetect/tf_JDAP/models/MTCNN_Official/det1.caffemodel'
caffe_net_file = '/home/dafu/workspace/FaceDetect/tf_JDAP/models/MTCNN_Official/det1.prototxt'
new_model_dir = './tf2caffe_rnet.caffemodel'
stage = 12
TFConverter = TensorflowParser(stage, tf_model_name, caffe_model_name)
tf_data = TFConverter.data

CaffeConverter = CaffeParser(caffe_net_file, caffe_model_name, tf_data)
CaffeConverter.tf2caffe()

