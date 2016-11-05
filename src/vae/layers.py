import tensorflow as tf
import numpy as np


def weight_initialization(fan_in, fan_out, constant=1, filter_size=list(),
                          inverse=False):
    """
    Xavier initialization of network weights.
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    initial_b = tf.zeros([fan_out])
    if inverse:
        initial_w = tf.random_uniform(
            list(filter_size) + [fan_out, fan_in], minval=low, maxval=high,
            dtype=tf.float32
        )
    else:
        initial_w = tf.random_uniform(
            list(filter_size) + [fan_in, fan_out], minval=low, maxval=high,
            dtype=tf.float32
        )
    return (tf.Variable(initial_w, trainable=True, name="weights"),
            tf.Variable(initial_b, trainable=True, name="biases"))


class FullyConnectedLayer:
    """
    Fully connected layer.
    """

    def __init__(self, input_size, output_size, scope="fc", dropout=1.,
                 activation=tf.nn.tanh):
        self.input_size = input_size
        self.output_size = output_size
        self.scope = scope
        self.dropout = dropout
        self.activation = activation
        with tf.name_scope(self.scope):
            self.w, self.b = weight_initialization(
                self.input_size, self.output_size
            )

    def __call__(self, x):
        h = self.activation(
            tf.add(
                tf.matmul(x, self.w), self.b
            )
        )
        h_dropout = tf.nn.dropout(h, self.dropout)
        return h_dropout


class ConvolutionalLayer:
    """
    Convolutional Layer
    """

    def __init__(self, input_size, output_size, scope="cl", dropout=1.,
                 activation=tf.nn.relu, stride=list([1, 1, 1, 1]),
                 filter_size=list([3, 3]), padding='SAME', inverse=False):
        self.scope = scope
        self.dropout = dropout
        self.activation = activation
        self.stride = stride
        self.filter_size = filter_size
        self.padding = padding
        self.inverse = inverse
        with tf.name_scope(self.scope):
            if not self.inverse:
                self.input_size = input_size
                self.output_size = output_size
                self.w, self.b = weight_initialization(
                    self.input_size, self.output_size,
                    filter_size=self.filter_size
                )
            else:
                self.input_size = input_size
                self.output_size = output_size
                self.w, self.b = weight_initialization(
                    self.input_size[-1], self.output_size[-1],
                    filter_size=self.filter_size, inverse=True
                )

    def __call__(self, x):
        if not self.inverse:
            h = self.activation(tf.add(
                tf.nn.conv2d(x, self.w, strides=self.stride,
                             padding=self.padding), self.b
            ))
            return tf.nn.dropout(h, self.dropout)
        else:
            output_shape = [x.get_shape()[0].value] + self.output_size[1:]
            h = self.activation(tf.add(
                tf.nn.conv2d_transpose(x, self.w, strides=self.stride,
                                       output_shape=output_shape,
                                       padding=self.padding), self.b
            ))
            return tf.nn.dropout(h, self.dropout)

class PoolingLayer:
    """
    Pooling Layer
    Depooling: https://gist.github.com/kastnerkyle/f3f67424adda343fef40
    https://github.com/pkmital/tensorflow_tutorials/blob/master/python/11_variational_autoencoder.py
    """

    def __init__(self, output_size, scope="pl", ksize=list([1, 2, 2, 1]),
                 stride=list([1, 2, 2, 1]),
                 padding='SAME', inverse=False):
        self.output_size = output_size
        self.scope = scope
        self.ksize = stride
        self.stride = stride
        self.padding = padding
        self.inverse = inverse

    def __call__(self, x):
        if not self.inverse:
            return tf.nn.max_pool(x, ksize=self.ksize, strides=self.stride,
                                  padding=self.padding)
        else:
            input_shape = x.get_shape().as_list()
            return tf.image.resize_images(x, (input_shape[1] * self.ksize[1],
                                          input_shape[2] * self.ksize[2]))
