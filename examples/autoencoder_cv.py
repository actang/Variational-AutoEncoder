import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from vae.autoencoder import AutoEncoder
import numpy as np

mnist = input_data.read_data_sets("./mnist_data")

model_architecture_cv = [
    {
        'layer': 'convolution',
        'layer_size': 64,
        'activation': tf.nn.relu,
        'stride': [1, 1, 1, 1],
        'padding': 'SAME',
    },
    {
        'layer': 'pooling',
        'layer_size': 64,
        'stride': [1, 1, 1, 1],
        'pooling_len': [1, 2, 2, 1],
        'padding': 'SAME',
    },
    {
        'layer': 'convolution',
        'layer_size': 128,
        'activation': tf.nn.relu,
        'stride': [1, 1, 1, 1],
        'padding': 'SAME',
    },
    {
        'layer': 'pooling',
        'layer_size': 128,
        'stride': [1, 1, 1, 1],
        'pooling_len': [1, 2, 2, 1],
        'padding': 'SAME',
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 512,
        'activation': tf.nn.elu,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 10,
        'activation': tf.nn.elu,
    },
]

model_architecture_fc = [
    {
        'layer': 'fullyconnected',
        'layer_size': 512,
        'activation': tf.nn.elu,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 256,
        'activation': tf.nn.elu,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 128,
        'activation': tf.nn.elu,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 64,
        'activation': tf.nn.elu,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 16,
        'activation': tf.nn.elu,
    },
]

v = AutoEncoder(input_size=[784],
                architecture=model_architecture_fc,
                batch_size=128,
                learning_rate=1e-3,
                dropout=1.0,
                l2_reg=1e-5,
                sesh=None,
                name='fc_mnist_autoencoder',
                )

v.train(mnist, max_iter=2 ** 15, max_epochs=np.inf, verbose=True, saver=True)
