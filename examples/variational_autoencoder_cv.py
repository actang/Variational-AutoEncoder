import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from vae.variationalautoencoder import VariationalAutoEncoder
import numpy as np
from vae.distribution import Distribution

mnist = input_data.read_data_sets("./mnist_data", reshape=False)

model_architecture_cv = [
    {
        'layer': 'convolution',
        'layer_size': 32,
        'activation': tf.nn.relu,
        'filter_size': [5, 5],
        'stride': [1, 1, 1, 1],
        'padding': 'SAME',
        'dropout': 1.0,
    },
    {
        'layer': 'convolution',
        'layer_size': 64,
        'activation': tf.nn.relu,
        'filter_size': [5, 5],
        'stride': [1, 1, 1, 1],
        'padding': 'SAME',
        'dropout': 1.0,
    },
    {
        'layer': 'pooling',
        'layer_size': 64,
        'stride': [1, 2, 2, 1],
        'pooling_len': [1, 2, 2, 1],
        'padding': 'SAME',
    },
    {
        'layer': 'convolution',
        'layer_size': 128,
        'activation': tf.nn.relu,
        'filter_size': [5, 5],
        'stride': [1, 1, 1, 1],
        'padding': 'SAME',
        'dropout': 0.9,
    },
    {
        'layer': 'pooling',
        'layer_size': 128,
        'stride': [1, 2, 2, 1],
        'pooling_len': [1, 2, 2, 1],
        'padding': 'SAME',
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 200,
        'activation': tf.identity,
        'dropout': 1.0,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 10,
        'activation': tf.identity,
        'dropout': 1.0,
    },
]

v = VariationalAutoEncoder(
    input_size=[28, 28, 1],
    architecture=model_architecture_cv,
    batch_size=100,
    distribution=Distribution("normal"),
    learning_rate=1e-5,
    l2_reg=1e-5,
    sesh=None,
    name='cv_mnist_variationalautoencoder',
)

v.train(mnist, max_iter=2 ** 16, max_epochs=np.inf, verbose=True, saver=True,
        plot_count=1000)
