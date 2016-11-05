import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from vae.autoencoder import AutoEncoder
import numpy as np

mnist = input_data.read_data_sets("./mnist_data", reshape=False)

model_architecture_cv = [
    {
        'layer': 'convolution',
        'layer_size': 8,
        'activation': tf.nn.relu,
        'filter_size': [3, 3],
        'stride': [1, 1, 1, 1],
        'padding': 'SAME',
        'dropout': 0.9,
    },
    {
        'layer': 'pooling',
        'layer_size': 8,
        'stride': [1, 2, 2, 1],
        'pooling_len': [1, 2, 2, 1],
        'padding': 'SAME',
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 200,
        'activation': tf.nn.elu,
        'dropout': 0.9,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 10,
        'activation': tf.nn.elu,
        'dropout': 0.9,
    },
]

v = AutoEncoder(
    input_size=[28, 28, 1],
    architecture=model_architecture_cv,
    batch_size=100,
    learning_rate=1e-5,
    l2_reg=1e-5,
    sesh=None,
    name='cv_mnist_autoencoder',
)

v.train(mnist, max_iter=2 ** 15, max_epochs=np.inf, verbose=True, saver=True)
