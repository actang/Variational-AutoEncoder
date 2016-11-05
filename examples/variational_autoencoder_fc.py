import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from vae.variationalautoencoder import VariationalAutoEncoder
import numpy as np
from vae.distribution import Distribution


mnist = input_data.read_data_sets("./mnist_data")

model_architecture_fc = [
    {
        'layer': 'fullyconnected',
        'layer_size': 256,
        'activation': tf.nn.sigmoid,
        'dropout': 0.9,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 128,
        'activation': tf.nn.sigmoid,
        'dropout': 0.9,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 64,
        'activation': tf.nn.softplus,
        'dropout': 0.9,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 16,
        'activation': tf.identity,
        'dropout': 1.0,
    },
]

v = VariationalAutoEncoder(
    input_size=[784],
    architecture=model_architecture_fc,
    batch_size=128,
    distribution=Distribution("normal"),
    learning_rate=5e-4,
    l2_reg=1e-5,
    sesh=None,
    name='fc_mnist_variationalautoencoder',
)

v.train(mnist, max_iter=2 ** 16, max_epochs=np.inf, verbose=True, saver=True,
        plot_count=1000)
