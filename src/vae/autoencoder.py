from vae.layers import layer_composition, FullyConnectedLayer
from vae.distribution import Distribution
from vae.loss import cross_entropy
from vae import plot
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys


class AutoEncoder:

    def __init__(self, architecture=None, save_model=True,
                 log_dir="log", batch_size=128, learning_rate=1e-3,
                 dropout=1., lambda_l2_reg=0., activation=tf.nn.elu,
                 squashing=tf.nn.sigmoid, distribution=Distribution("normal")):
        """
        Build up a symmetric autoencoder.

        Currently only support fully connected layer, in the future
        architecture can be changed to a list of tuples, which stores the
        layer type and neuron amount. For example right now [1000, 500, 250, 10]
        specifies an autoencoder with 1000-D inputs, 10-D latents, & end-to-end
        architecture [1000, 500, 250, 10, 250, 500, 1000].
        """

        # Assign parameters to the object
        self.architecture = architecture
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = tf.placeholder_with_default(dropout,
                                                   shape=[],
                                                   name="dropout")
        self.lambda_l2_reg = lambda_l2_reg
        self.activation = activation
        self.squashing = squashing
        self.distribution = distribution

        # Initialize first Tensorflow object
        self.x_reconstructed = None
        self.x_in = None
        self.global_step = None
        self.cost = None
        self.train_op = None
        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        # Start a Tensorflow session
        self.sesh = tf.Session()

        # Tensorboard
        if save_model:
            self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)

    def build_encoder(self):
        """
        :return: encoder function
        """
        # Input data x
        self.x_in = tf.placeholder(tf.float32,
                                   shape=[None, self.architecture[0]],
                                   name="x")

        # Encoding / "recognition" {q(z|x)}
        # Hidden layers reversed for function composition
        encoding = [FullyConnectedLayer(size=hidden_size,
                                        scope="encoding",
                                        dropout=self.dropout,
                                        activation=self.activation)
                    for hidden_size in reversed(self.architecture[1:-1])]
        encoder = layer_composition(encoding)(self.x_in)
        return encoder

    def build_decoder(self):
        """
        :return: decoder function
        """
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
