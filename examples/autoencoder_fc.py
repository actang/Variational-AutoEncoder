import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from vae.autoencoder import AutoEncoder
import numpy as np

mnist = input_data.read_data_sets("./mnist_data")

model_architecture_fc = [
    {
        'layer': 'fullyconnected',
        'layer_size': 500,
        'activation': tf.nn.elu,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 500,
        'activation': tf.nn.elu,
    },
    {
        'layer': 'fullyconnected',
        'layer_size': 20,
        'activation': tf.nn.elu,
    },
]

v = AutoEncoder(input_size=[784],
                architecture=model_architecture_fc,
                batch_size=128,
                learning_rate=0.001,
                dropout=0.9,
                l2_reg=1e-10,
                sesh=None,
                name='fc_mnist_autoencoder',
                )

v.train(mnist, max_iter=2 ** 12, max_epochs=np.inf, verbose=True, saver=True,
        plot_count=100)
