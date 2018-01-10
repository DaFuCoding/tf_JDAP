# Quantization analysis network weights and output
import tensorflow as tf
import collections

class Quantizer(object):
    def __init__(self):
        self._layer_names = []
        # Max value of in/outputs, param per layer
        self._max_in = []
        self._max_out = []
        self._max_param = []
        # The integer length for dynamic fixed point
        self._il_in = []
        self._il_out = []
        self._il_param = []
        self._base_bitwidth = 16

    def ActivationAnalysis(self):
        pass
    def WeightAnalysis(self, is_per_layer=False):
        pass
    def BitWidthReduction(self):
        pass
    def Quantize2DynamicFixedPoint(self):
        pass


