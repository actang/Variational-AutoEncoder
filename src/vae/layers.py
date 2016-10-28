import tensorflow as tf


def weight_initialization(fan_in, fan_out, filter_size=list()):
    """
    Initialize weights and biases.

    :param fan_in: Input data dimension.
    :param fan_out: Output data dimension (layer size).
    :param filter_size: filter size dimension (for convolution and pooling).
    :return: Randomly initialized variables for w and b
    """
    stddev = tf.cast((2 / fan_in) ** 0.5, tf.float32)
    initial_w = tf.random_normal(list(filter_size) + [fan_in, fan_out],
                                 stddev=stddev)
    initial_b = tf.zeros([fan_out])
    return (tf.Variable(initial_w, trainable=True, name="weights"),
            tf.Variable(initial_b, trainable=True, name="biases"))


class FullyConnectedLayer:
    """
    Fully connected layer.
    """

    def __init__(self, size, scope="fc", dropout=1., activation=tf.nn.tanh):
        self.size = size
        self.scope = scope
        self.dropout = dropout
        self.activation = activation

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                try:
                    h = self.activation(
                        tf.nn.bias_add(
                            tf.matmul(x, self.w), self.b
                        )
                    )
                    return tf.nn.dropout(h, self.dropout)
                except AttributeError:
                    self.w, self.b = weight_initialization(
                        x.get_shape()[1].value, self.size
                    )


class ConvolutionalLayer:
    """
    Convolutional Layer
    """

    def __init__(self, size, scope="cl", dropout=1.0, activation=tf.nn.relu,
                 stride=None, padding='SAME', output_size=None, inverse=False):
        self.size = size
        self.scope = scope
        self.dropout = dropout
        self.activation = activation
        if stride is None:
            self.stride = [1, 1, 1, 1]
        else:
            self.stride = stride
        self.padding = padding
        if output_size is not None:
            self.output_size = output_size[1:]
        else:
            self.output_size = output_size
        self.inverse = inverse

    def __call__(self, x):
        with tf.name_scope(self.scope):
            while True:
                try:
                    if not self.inverse:
                        h = self.activation(
                            tf.nn.bias_add(
                                tf.nn.conv2d(
                                    x, self.w, strides=self.stride,
                                    padding=self.padding
                                ), self.b
                            )
                        )
                        return tf.nn.dropout(h, self.dropout)
                    else:
                        h = self.activation(
                            tf.nn.bias_add(
                                tf.nn.conv2d_transpose(
                                    x, self.w,
                                    output_shape=[x.get_shape()[0].value] +
                                                 self.output_size,
                                    strides=self.stride,
                                    padding=self.padding
                                ), self.b
                            )
                        )
                        return tf.nn.dropout(h, self.dropout)
                except AttributeError:
                    input_size = x.get_shape()[-1].value
                    filter_size = [i.value for i in x.get_shape()[1: -1]]
                    self.w, self.b = weight_initialization(
                        input_size, self.size, filter_size
                    )


class PoolingLayer:
    """
    Pooling Layer
    Depooling: https://gist.github.com/kastnerkyle/f3f67424adda343fef40
    https://github.com/pkmital/tensorflow_tutorials/blob/master/python/11_variational_autoencoder.py
    """

    def __init__(self, size, scope="pl", ksize=None, stride=None,
                 padding='SAME', inverse=False):
        self.size = size
        self.scope = scope
        if ksize is None:
            self.ksize = [1, 2, 2, 1]
        else:
            self.ksize = stride
        if stride is None:
            self.stride = [1, 1, 1, 1]
        else:
            self.stride = stride
        self.padding = padding
        self.inverse = inverse

    def __call__(self, x):
        if not self.inverse:
            return tf.nn.max_pool(x, ksize=self.ksize, strides=self.stride,
                                  padding=self.padding)
        else:
            input_shape = x.get_shape().as_list()
            return tf.image.resize_images(x, input_shape[1] * self.ksize[1],
                                          input_shape[2] * self.ksize[2])
