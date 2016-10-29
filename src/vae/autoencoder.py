from vae.layers import FullyConnectedLayer, ConvolutionalLayer, PoolingLayer
from vae.loss import cross_entropy
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys


class AutoEncoder:
    LOG_FOLDER_NAME = 'logs'
    MODEL_FOLDER_NAME = 'models'
    PLOT_FOLDER_NAME = 'plots'

    def __init__(self, input_size, architecture, batch_size=128,
                 learning_rate=1e-3, dropout=1.0, l2_reg=1e-5, sesh=None,
                 name="autoencoder"):
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
            [
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
        :param batch_size: Batch size for each training cycle.
        :param learning_rate: Learning rate for updating gradients.
        :param dropout: 1 - probability of dropout (1 means no dropout).
        :param l2_reg: l2 regularization lambda.
        :param sesh: A preloaded tensorflow session or None.
        :param name: A customized name for the object
        """
        self.input_size = input_size
        self.architecture = architecture
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name = name
        self.dropout = tf.placeholder_with_default(dropout, shape=[],
                                                   name="dropout")
        self.lambda_l2_reg = l2_reg

        # Tensorflow variable: reconstructed images
        self.x_reconstructed = None
        # Tensorflow variable: Original images
        self.x = tf.placeholder(tf.float32,
                                shape=[None] + list(input_size),
                                name="x")
        # Tensorflow variable: Training operation applying specified gradients
        self.training_operation = None
        # Tensorflow variable: Loss function
        self.cost = None
        # Global training step
        self.global_step = 0
        # Store a list of the input and output shapes of encoder layers
        self.shapes = []

        # Store current dateime
        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        # Start a Tensorflow session if not
        if sesh is None:
            self.sesh = tf.Session()
        else:
            self.sesh = sesh

        # Save to SummaryWriter for Tensorboard
        self.logger = tf.train.SummaryWriter(self.LOG_FOLDER_NAME,
                                             self.sesh.graph)
        self._build_graph()

    def encoder(self):
        """
        Build a aggregate function for encoder.

        :return: encoder function
        """
        current_input = self.x

        for i, layer in enumerate(self.architecture):

            input_size = current_input.get_shape().as_list()

            current_layer = None
            if layer['layer'] == "convolution":
                current_layer = ConvolutionalLayer(
                    size=layer['layer_size'],
                    scope="convolution_layer_encoder_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    inverse=False
                )
            elif layer['layer'] == "pooling":
                current_layer = PoolingLayer(
                    size=layer['layer_size'],
                    scope="pooling_layer_encoder_{0}".format(i),
                    ksize=layer['pooling_len'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    inverse=False
                )
            elif layer['layer'] == "fullyconnected":
                current_layer = FullyConnectedLayer(
                    size=layer['layer_size'],
                    scope="fully_connected_layer_encoder_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation']
                )
            else:
                pass
            output_size = current_input.get_shape().as_list()
            self.shapes.append((input_size, output_size))

            # flatten
            if layer['layer'] == "fullyconnected" and i > 1 \
                    and self.architecture[i - 1]['layer'] != "fullyconnected":
                batch_size = current_input.get_shape()[0].value
                current_input = tf.reshape(current_input, [batch_size, -1])

            current_input = current_layer(current_input)
        return current_input

    def decoder(self, z):
        """
        Build an aggregate function for decoder.

        :return: decoder function
        """
        current_output = z

        for i in reversed(range(len(self.architecture))):
            layer = self.architecture[i]
            input_size = self.shapes[i][0]
            output_size = self.shapes[i][1]
            current_layer = None

            if layer['layer'] == "convolution":
                current_layer = ConvolutionalLayer(
                    size=input_size[-1],
                    scope="convolution_layer_decoder_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    output_size=output_size,
                    inverse=True
                )

            elif layer['layer'] == "pooling":
                current_layer = PoolingLayer(
                    size=input_size[-1],
                    scope="pooling_layer_decoder_{0}".format(i),
                    ksize=layer['pooling_len'],
                    stride=layer['stride'],
                    padding=layer['padding'],
                    inverse=True
                )

            elif layer['layer'] == "fullyconnected":
                current_layer = FullyConnectedLayer(
                    size=input_size[-1],
                    scope="fully_connected_layer_decoder_{0}".format(i),
                    dropout=self.dropout,
                    activation=layer['activation']
                )
            else:
                pass

            # Unflatten
            if layer['layer'] != "fullyconnected" and \
                i < len(self.architecture) - 1 and \
                    self.architecture[i + 1]['layer'] == "fullyconnected":
                batch_size = current_output.get_shape()[0].value
                filter_size = output_size[1:]
                current_output = tf.reshape(current_output,
                                            [batch_size] + filter_size)

            current_output = current_layer(current_output)

        return current_output

    # def reconstruct_x(self, z):
    #     # Reconstruct the object from the latent variables
    #     self.x_reconstructed = tf.identity(self.decoder(z),
    #                                        name="x_reconstructed")

    def reconstruct_loss(self, x_reconstructed):
        # Calculate reconstruction loss
        rec_loss = cross_entropy(x_reconstructed, self.x)
        return rec_loss

    def _build_graph(self):
        latent = self.encoder()
        self.x_reconstructed = tf.identity(self.decoder(latent),
                                           name="x_reconstructed")
        self.calculate_cost(self.x_reconstructed)
        self.optimization_function()
        self.sesh.run(tf.initialize_all_variables())

    # def latent_gradient(self, z, image_grad):
    #     # compute gradients for manifold exploration
    #     z_grad = tf.gradients(self.reconstruct_x(z), [z],
    #                           grad_ys=image_grad)[0]
    #     return z_grad

    def calculate_cost(self, x_reconstructed):
        with tf.name_scope("l2_regularization"):
            regularizers = [
                tf.nn.l2_loss(var) for var in
                self.sesh.graph.get_collection("trainable_variables") if
                "weights" in var.name
                ]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)
        with tf.name_scope("total_cost"):
            # Average over minibatch
            self.cost = tf.reduce_mean(self.reconstruct_loss(x_reconstructed),
                                       name="vae_cost")
            self.cost += l2_reg

    def optimization_function(self):
        self.global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            var_list = tf.trainable_variables()
            # Calculate gradient loss
            grads_and_vars = optimizer.compute_gradients(self.cost, var_list)
            # Gradient clipping
            grads_and_vars = [
                (tf.clip_by_value(grad, -5, 5), tvar)
                for grad, tvar in grads_and_vars
            ]
            self.training_operation = optimizer.apply_gradients(
                grads_and_vars, self.global_step, name="minimize_cost"
            )

    @property
    def step(self):
        return self.global_step.eval(session=self.sesh)

    def plot_sample(self, x, x_reconstructed, sample=10, columns=None,
                    name="training_sample", outdir=PLOT_FOLDER_NAME):
        sample = min(sample, x.shape[0])
        columns = columns if columns else sample
        rows = 2 * int(np.ceil(sample / columns))
        plt.figure(figsize=(columns * 2, rows * 2))
        # Assume it's a square image
        if len(self.input_size) == 1:
            dim = int(self.input_size[0] ** 0.5)
            dim = [dim, dim]
        elif len(self.input_size) >= 2:
            dim = self.input_size

        def draw_subplot(x_, ax_):
            plt.imshow(x_.reshape(dim), cmap="Greys")
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)

        for i, x in enumerate(x[:sample], 1):
            ax = plt.subplot(rows, columns, i)
            draw_subplot(x, ax)

        for i, x in enumerate(x_reconstructed[:sample], 1):
            ax = plt.subplot(rows, columns, i + columns * (rows / 2))
            draw_subplot(x, ax)

        file_name = "{0}_{1}_batch_sample_round_{2}_{3}.png".format(
            self.datetime, self.name, self.step, name
        )
        plt.savefig(os.path.join(outdir, file_name), bbox_inches="tight")

    def train(self, train_x, max_iter=np.inf, max_epochs=np.inf,
              verbose=True, saver=True, plot_count=1000):

        if saver:
            saver = tf.train.Saver(tf.all_variables())
        else:
            saver = None

        print("---> Autoencoder structure:")
        for i, layer in enumerate(self.architecture):
            print("Layer {0}: {1} with size {2}".format(
                i, layer['layer'], layer['layer_size'])
            )

        training_error = 0
        iteration = 1

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            "---> Autoencoder training starts at {0}.".format(current_time)
        )

        try:
            while True:
                # x is a batch of images, _ is a batch of their labels(ignored)
                x, _ = train_x.train.next_batch(self.batch_size)
                feed_dict = {self.x: x}
                fetches = [self.x_reconstructed, self.cost,
                           self.global_step, self.training_operation]
                x_reconstructed, cost, iteration, _ = self.sesh.run(fetches,
                                                                    feed_dict)
                training_error += cost

                # if plot_latent_over_time is not None:
                #     while int(round(BASE ** pow_)) == i:
                #         nnplot.exploreLatent(self, nx=30, ny=30,
                #                              ppf=True, outdir=plots_outdir,
                #                              name="explore_ppf30_{}".format(pow_))
                #         names = ("train", "validation", "test")
                #         datasets = (train_x.train, train_x.validation,
                #                     train_x.test)
                #         for name, dataset in zip(names, datasets):
                #             nnplot.plotInLatent(self, dataset.images,
                #                                 dataset.labels,
                #                                 range_=(-6, 6), title=name,
                #                                 outdir=plots_outdir,
                #                                 name="{}_{}".format(name, pow_))
                #         print("{}^{} = {}".format(BASE, pow_, i))
                #         pow_ += INCREMENT

                if verbose and iteration % plot_count == 0:
                    print("round {0}: average cost is {1}.".format(
                        iteration, training_error / iteration))
                    self.plot_sample(x, x_reconstructed, sample=10, columns=5,
                                     name="training",
                                     outdir=self.PLOT_FOLDER_NAME)
                    x, _ = train_x.validation.next_batch(self.batch_size)
                    feed_dict = {self.x: x}
                    fetches = [self.x_reconstructed, self.cost]
                    x_reconstructed, cost = self.sesh.run(fetches, feed_dict)
                    print("round {0}: testing cost is {1}.".format(
                        iteration, cost))
                    self.plot_sample(x, x_reconstructed, sample=10, columns=5,
                                     name="testing",
                                     outdir=self.PLOT_FOLDER_NAME)

                if iteration >= max_iter or \
                        train_x.train.epochs_completed >= max_epochs:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(
                        "---> Autoencoder training ends at {0}.".format(
                            current_time)
                    )
                    print("Total steps: {0}".format(iteration))
                    print("Total epochs: {0}".format(
                        train_x.train.epochs_completed))
                    print("Average cost: {0}".format(
                        training_error / iteration))

                    if saver is not None:
                        output_file = os.path.join(
                            os.path.abspath(self.MODEL_FOLDER_NAME),
                            "{0}_{1}".format(self.datetime, self.name))
                        saver.save(self.sesh, output_file,
                                   global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except AttributeError:
                        pass

                    break

        except KeyboardInterrupt:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                "---> Autoencoder training interrupted at {0}.".format(
                    current_time)
            )
            print("Total steps: {0}".format(iteration))
            print("Total epochs: {0}".format(
                train_x.train.epochs_completed))
            print("Average cost: {0}".format(training_error / iteration))
            sys.exit(0)
