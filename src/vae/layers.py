import functools
from functional import compose, partial
import tensorflow as tf


def layer_composition(*args):
    """Util for multiple function composition

    i.e. composed = composeAll([f, g, h])
         composed(x) # == f(g(h(x)))
    """
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)


def weight_initialization(fan_in: int, fan_out: int):
    """
    Helper to initialize weights and biases, via He's adaptation
    of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
    """
    # (int, int) -> (tf.Variable, tf.Variable)
    stddev = tf.cast((2 / fan_in) ** 0.5, tf.float32)
    initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
    initial_b = tf.zeros([fan_out])
    return (tf.Variable(initial_w, trainable=True, name="weights"),
            tf.Variable(initial_b, trainable=True, name="biases"))


class FullyConnectedLayer:
    """
    Fully connected layer.
    Based on http://blog.fastforwardlabs.com/post/149329060653/under-the-hood-
    of-the-variational-autoencoder-in
    """
    def __init__(self,
                 size,  # number of neurons on the layer
                 scope="fc",  # name for the layer type
                 dropout=1.,  # 1 - probability of dropout (1 means no dropout)
                 activation=tf.nn.elu  # non-linearity activation function
                 ):
        self.size = size
        self.scope = scope
        self.dropout = dropout  # keep_prob
        self.activation = activation

    def __call__(self, x):
        # Dense layer currying, to apply layer to any input tensor x
        with tf.name_scope(self.scope):
            while True:
                try:
                    # reuse weights if already initialized
                    h = self.activation(tf.matmul(x, self.w) + self.b)
                except AttributeError:
                    self.w, self.b = weight_initialization(
                        x.get_shape()[1].value, self.size
                    )


class ConvolutionalLayer:
    """
    Convolutional Layer
    """
    def __init__(self,
                 size,  # number of neurons on the layer
                 scope="cl",  # name for the layer type
                 dropout=1.,  # 1 - probability of dropout (1 means no dropout)
                 activation=tf.nn.relu,  # non-linearity activation function,
                 stride=[1, 1, 1, 1],
                 padding='SAME',
                 decov=False
                 ):
        self.size = size
        self.scope = scope
        self.dropout = dropout  # keep_prob
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.decov = decov

    def __call__(self, x):
        # Dense layer currying, to apply layer to any input tensor x
        with tf.name_scope(self.scope):
            while True:
                try:
                    if not self.decov:
                        # reuse weights if already initialized
                        h = self.activation(
                            tf.nn.conv2d(
                                x, self.w, strides=self.stride,
                                padding=self.padding)
                            + self.b
                        )
                        return tf.nn.dropout(h, self.dropout)
                    else:
                        h = self.activation(
                            tf.nn.conv2d_transpose(
                                x, self.w, strides=self.stride,
                                padding=self.padding)
                            + self.b
                        )
                        return tf.nn.dropout(h, self.dropout)
                except AttributeError:
                    self.w, self.b = weight_initialization(
                        x.get_shape()[1].value, self.size
                    )


class PoolingLayer:
    """
    Pooling Layer
    Depooling: https://gist.github.com/kastnerkyle/f3f67424adda343fef40
    """
    def __init__(self,
                 size,  # number of neurons on the layer
                 scope="pl",  # name for the layer type
                 ksize=[1, 2, 2, 1],
                 stride=[1, 1, 1, 1],
                 padding='SAME',
                 decov=False
                 ):
        self.size = size
        self.scope = scope
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.decov = decov

    def __call__(self, x):
        if not self.decov:
            return tf.nn.max_pool(x, ksize=self.ksize,
                                  strides=self.stride, padding=self.padding)
        else:
            return
