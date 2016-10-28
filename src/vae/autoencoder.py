from vae.layers import FullyConnectedLayer, ConvolutionalLayer, PoolingLayer
from vae.loss import cross_entropy
from vae import plot
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys


class AutoEncoder:

    LOG_FOLDER_NAME = 'logs'
    MODEL_FOLDER_NAME = 'models'

    def __init__(self, input_size, architecture, batch_size=128,
                 learning_rate=1e-3, dropout=1.0, l2_reg=1e-5, sesh=None,
                 save_model=True):
        """
        Initialize a symmetric autoencoder.

        :param input_size: A tuple of the dimension of the input image.
        :param architecture: a list of dictionaries containing layer info.
        It currently supports three different kinds of layers: 'convolution',
        'fullyconnected' and 'pooling'. For each type of layer, 'layer_size'
        is required. For 'convolution', it is necessary to provide
        'activation', 'layer_size', 'stride' and 'padding'. For 'pooling',
        it is also necessary to provide 'stride' and 'padding' and
        'pooling_len'. 'layer_size' will simply follow the last
        convolution layer before the pooling layer. For 'fullyconnected',
        it is also necessary to provide 'activation'. Please note that this
        architecture list should only be the encoder's. It will automatically
        generate the symmetric architecture for the decoder. An example is
        shown below:
             [{
                'layer' : 'convolution',
                'layer_size' : 64,
                'activation' : tf.nn.relu,
                'stride' : [1, 1, 1, 1],
                'padding' : 'SAME',
              },
              {
                'layer' : 'pooling',
                'layer_size' : 64,
                'stride' : [1, 1, 1, 1],
                'pooling_len' : [1, 2, 2, 1],
                'padding' : 'SAME',
              },
              {
                'layer' : 'convolution',
                'layer_size' : 128,
                'activation' : tf.nn.relu,
                'stride' : [1, 1, 1, 1],
                'padding' : 'SAME',
              },
              {
                'layer' : 'pooling',
                'layer_size' : 128,
                'stride' : [1, 1, 1, 1],
                'pooling_len' : [1, 2, 2, 1],
                'padding' : 'SAME',
              },
              {
                'layer' : 'fullyconnected',
                'layer_size' : 512,
                'activation' : tf.nn.elu,
               },
              {
                'layer' : 'fullyconnected',
                'layer_size' : 10,
                'activation' : tf.nn.elu,
               },
             ]
        :param batch_size: Batch size for each training cycle.
        :param learning_rate: Learning rate for updating gradients.
        :param dropout: 1 - probability of dropout (1 means no dropout).
        :param l2_reg: l2 regularization lambda.
        :param sesh: A preloaded tensorflow session or None.
        :param save_model: If True, will save logs and models to disk.
        """

        self.architecture = architecture
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = tf.placeholder_with_default(dropout, shape=[],
                                                   name="dropout")
        self.lambda_l2_reg = l2_reg
        self.save_model = save_model

        # Initialize first Tensorflow object
        self.x_reconstructed = None
        self.x = tf.placeholder(tf.float32,
                                shape=[None] + list(input_size),
                                name="x")
        self.global_step = None
        self.cost = None
        self.train_op = None

        # Store current dateime
        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        # Start a Tensorflow session if not
        if self.sesh is None:
            self.sesh = tf.Session()
        else:
            self.sesh = sesh

        # Save to SummaryWriter for Tensorboard
        if self.save_model:
            self.logger = tf.train.SummaryWriter(self.LOG_FOLDER_NAME,
                                                 self.sesh.graph)

    def encoder(self):
        """
        Build a aggregate function for encoder.

        :return: encoder function
        """
        current_input = self.x

        for i, layer in enumerate(self.architecture):
            current_layer = None
            if layer['layer'] == "convolution":
                current_layer = ConvolutionalLayer(
                    size=layer['layer_size'],
                    scope="convolution_layer_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    inverse=False
                )
            elif layer['layer'] == "pooling":
                current_layer = PoolingLayer(
                    size=layer['layer_size'],
                    scope="pooling_layer_{0}".format(i),
                    ksize=layer['pooling_len'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    inverse=False
                )
            elif layer['layer'] == "fullyconnected":
                current_layer = FullyConnectedLayer(
                    size=layer['layer_size'],
                    scope="fully_connected_layer_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation']
                )
                batch_size = current_input.get_shape()[0].value
                current_input = tf.reshape(current_input, [batch_size, -1])
            else:
                pass
            current_input = current_layer(current_input)

        return current_input

    def build_decoder(self, z):
        """
        Build an aggregate function for decoder.

        :return: decoder function
        """
        current_input = z
        last_conv_size = None

        for i, layer in enumerate(reversed(self.architecture)):
            current_layer = None
            if layer['layer'] == "convolution":
                current_layer = ConvolutionalLayer(
                    size=layer['layer_size'],
                    scope="convolution_layer_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    inverse=True
                )
            elif layer['layer'] == "pooling":
                current_layer = PoolingLayer(
                    size=layer['layer_size'],
                    scope="pooling_layer_{0}".format(i),
                    ksize=layer['pooling_len'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    inverse=True
                )
            elif layer['layer'] == "fullyconnected":
                current_layer = FullyConnectedLayer(
                    size=layer['layer_size'],
                    scope="fully_connected_layer_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation']
                )
                batch_size = current_input.get_shape()[0].value
                current_input = tf.reshape(current_input, [batch_size, -1])
            else:
                pass
            current_input, cache = current_layer(current_input)

        return current_input
        # Decoding / "generative": {p(x|z)}
        decoding = [FullyConnectedLayer(size=hidden_size,
                                        scope="decoding",
                                        dropout=self.dropout,
                                        activation=self.activation)
                    for hidden_size in self.architecture[1:-1]]

        # Restore original dimensions, squash outputs [0, 1]
        decoding.insert(0,
                        FullyConnectedLayer(
                            size=self.architecture[0],
                            scope="x_decoding",
                            dropout=self.dropout,
                            activation=self.squashing
                        ))
        decoder = layer_composition(decoding)
        return decoder

    def reconstruct_x(self, decoder, z):
        # Reconstruct the object from the latent variables
        self.x_reconstructed = tf.identity(decoder(z), name="x_reconstructed")

    def reconstruct_loss(self):
        # Calculate reconstruction loss
        rec_loss = cross_entropy(self.x_reconstructed, self.x_in)
        return rec_loss

    def _build_graph(self):
        pass

    def latent_gradient(self, z, image_grad):
        # compute gradients for manifold exploration
        z_grad = tf.gradients(self.decoder(z), [z], grad_ys=image_grad)[0]
        return z_grad

    def calculate_cost(self):
        with tf.name_scope("l2_regularization"):
            regularizers = [
                tf.nn.l2_loss(var) for var in
                self.sesh.graph.get_collection("trainable_variables") if
                "weights" in var.name
            ]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # average over minibatch
            self.cost = tf.reduce_mean(
                self.reconstruct_loss(),
                name="vae_cost")
            self.cost += l2_reg

    def optimization_function(self):
        self.global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(self.cost, tvars)
            # Gradient clipping
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar)
                       for grad, tvar in grads_and_vars]
            self.train_op = optimizer.apply_gradients(
                clipped,
                global_step=self.global_step,
                name="minimize_cost"
            )

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def train(self, train_x, max_iter=np.inf, max_epochs=np.inf,
              cross_validate=True, verbose=True, saver=True,
              outdir="models", plots_outdir="plots",
              plot_latent_over_time=False):

        if saver:
            saver = tf.train.Saver(tf.all_variables())
        else:
            saver = None

        err_train = 0
        i = 0
        try:
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            BASE = 2
            INCREMENT = 0.5
            pow_ = 0

            while True:
                x, _ = train_x.train.next_batch(self.batch_size)
                feed_dict = {self.x_in: x}
                fetches = [self.x_reconstructed, self.cost, self.global_step,
                           self.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)
                err_train += cost

                if plot_latent_over_time:
                    while int(round(BASE ** pow_)) == i:
                        plot.exploreLatent(self, nx=30, ny=30,
                                           ppf=True, outdir=plots_outdir,
                                           name="explore_ppf30_{}".format(pow_))
                        names = ("train", "validation", "test")
                        datasets = (train_x.train, train_x.validation,
                                    train_x.test)
                        for name, dataset in zip(names, datasets):
                            plot.plotInLatent(self, dataset.images,
                                              dataset.labels,
                                              range_=(-6, 6), title=name,
                                              outdir=plots_outdir,
                                              name="{}_{}".format(name, pow_))
                        print("{}^{} = {}".format(BASE, pow_, i))
                        pow_ += INCREMENT

                if i % 1000 == 0 and verbose:
                    print("round {} --> avg cost: ".format(i), err_train / i)

                if i % 2000 == 0 and verbose:  # and i >= 10000:
                    # visualize `n` examples of current minibatch
                    # inputs + reconstructions
                    plot.plotSubset(self, x, x_reconstructed, n=10,
                                    name="train", outdir=plots_outdir)

                    if cross_validate:
                        x, _ = train_x.validation.next_batch(
                            self.batch_size)
                        feed_dict = {self.x_in: x}
                        fetches = [self.x_reconstructed,
                                   self.cost]
                        x_reconstructed, cost = self.sesh.run(fetches,
                                                              feed_dict)

                        print("round {} --> CV cost: ".format(i), cost)
                        plot.plotSubset(self, x, x_reconstructed,
                                        n=10, name="cv", outdir=plots_outdir)

                if i >= max_iter \
                        or train_x.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, train_x.train.epochs_completed, err_train / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    if saver is not None:
                        outfile = os.path.join(
                            os.path.abspath(outdir),
                            "{}_vae_{}".format(self.datetime,
                                               "_".join(map(str,
                                                            self.architecture)
                                                        )
                                               )
                        )
                        saver.save(self.sesh, outfile, global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except AttributeError:  # not logging
                        continue
                    break

        except KeyboardInterrupt:
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, train_x.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)
