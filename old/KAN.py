import tensorflow as tf
import numpy as np
from scipy.interpolate import BSpline
from scipy import signal

"""
Personal attempt at KAN implementation

For experimental purposes in identifying how to fuse kans

RNN-KAN mixture model
KAN will utilize a b-spline activation function wrapped in a log-wavelet transform
RNN used to do back-propogation and process sequential data
"""

# B-Spline Layer
class BSplineLayer(tf.keras.Layer):
    def __init__(self, knots, degree = 3, **kwargs):
        super(BSplineLayer, self).__init__(**kwargs)
        self.knots = knots
        self.degree = degree

    def call(self, inputs):
        bsplines = []
        for i in range(inputs.shape[1]):
            b_spline = BSpline(self.knots, inputs[:, i], self.degree)
            bsplines.append(b_spline[inputs[:, i]])
        return tf.convert_to_tensor(bsplines, dtype = tf.float32)