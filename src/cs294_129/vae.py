from datetime import datetime
import os
import re
import tensorflow as tf
from cs294_129.layers import FullyConnectedLayer
from cs294_129.distribution import Distribution
from cs294_129.autoencoder import AutoEncoder


class VariationalAutoEncoder(AutoEncoder):
    """
    Variational Autoencoder
s
    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    def __init__(self, architecture=None, load_model=None, save_model=True,
                 log_dir="./log", batch_size=128, learning_rate=1e-3,
                 dropout=1., lambda_l2_reg=0., activation=tf.nn.elu,
                 squashing=tf.nn.sigmoid, distribution=Distribution("normal")):
        # Inherit autoencoder
        AutoEncoder.__init__(self, architecture=architecture,
                             save_model=save_model, log_dir=log_dir,
                             batch_size=batch_size, learning_rate=learning_rate,
                             dropout=dropout, lambda_l2_reg=lambda_l2_reg,
                             activation=activation, squashing=squashing,
                             distribution=distribution)

        # Initialize latent space distribution
        self.z_mean = None
        self.z_log_sigma = None
        self.x_reconstructed_ = None
        self.z_ = None

        for handle in (self.x_in, self.dropout, self.z_mean,
                       self.z_log_sigma, self.x_reconstructed, self.z_,
                       self.x_reconstructed_, self.cost,
                       self.global_step, self.train_op):
            tf.add_to_collection("vae_parameters", handle)

        if load_model is not None:
            # Load existing model
            model_datetime, model_name = os.path.basename(load_model).\
                split("_vae_")
            self.datetime = "{}_reloaded".format(model_datetime)
            *model_architecture, _ = re.split("_|-", model_name)
            self.architecture = [int(n) for n in model_architecture]
            meta_graph = os.path.abspath(load_model)
            tf.train.import_meta_graph(meta_graph + ".meta").\
                restore(self.sesh, meta_graph)
            (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
             self.x_reconstructed, self.z_, self.x_reconstructed_,
             self.cost, self.global_step,
             self.train_op) = self.sesh.graph.get_collection("vae_parameters")
        else:
            # Train a new graphical model
            self._build_graph()
            self.sesh.run(tf.initialize_all_variables())

    def _build_conjugate_latent_space(self):
        # Defaults to prior z ~ N(0, I)
        with tf.name_scope("latent_in"):
            self.z_ = tf.placeholder_with_default(
                tf.random_normal([1, self.architecture[-1]]),
                shape=[None, self.architecture[-1]],
                name="latent_in")

    def reconstruct_x_conjugate(self, decoder):
        # Reconstruct the object from the latent variables
        self.x_reconstructed_ = tf.identity(decoder(self.z_),
                                            name="x_reconstructed_")

    def _build_graph(self):
        encoder = self.build_encoder()
        self._build_latent_space(encoder)
        self._build_conjugate_latent_space()
        z = self.sample_latent_space(self.z_mean, self.z_log_sigma)
        decoder = self.build_decoder()
        self.reconstruct_x(decoder, z)
        self.reconstruct_x_conjugate(decoder)
        self.calculate_cost_with_dl_divergence(self.z_mean, self.z_log_sigma)
        self.optimization_function()

    def _build_latent_space(self, encoder):
        # Latent variable distribution is a multivariate normal with mean
        # z_mean and diagonal covariance z_log_sigma
        self.z_mean = FullyConnectedLayer(size=self.architecture[-1],
                                          scope="latent",
                                          dropout=self.dropout,
                                          activation=tf.identity)(encoder)
        self.z_log_sigma = FullyConnectedLayer(size=self.architecture[-1],
                                               scope="z_log_sigma",
                                               dropout=self.dropout,
                                               activation=tf.identity)(encoder)

    def sample_latent_space(self, z_mean, z_log_sigma):
        # Sample the latent distribution. Only one draw is necessary as long as
        # minibatch large enough (>100)
        z = self.distribution.sample_distribution(z_mean, z_log_sigma)
        return z

    def calulate_kl_divergence(self, z_mean, z_log_sigma):
        # Kullback-Leibler divergence
        kl_loss = self.distribution.kl_divergence(z_mean, z_log_sigma)
        return kl_loss

    def encode(self, x):
        """
        Probabilistic encoder from inputs to latent distribution parameters.
        {inference network q(z|x)}
        """
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_log_sigma],
                             feed_dict=feed_dict)

    def decode(self, zs=None):
        """
        Generative decoder from latent space to reconstructions of input.
        space {generative network p(x|z)}
        """
        feed_dict = dict()
        if zs is not None:
            # Coerce to np.array
            zs = (self.sesh.run(zs) if hasattr(zs, "eval") else zs)
            feed_dict.update({self.z_: zs})
        # Else zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sesh.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        """
        End-to-end autoencoder.
        """
        return self.decode(self.sample_latent_space(*self.encode(x)))

    def calculate_cost_with_dl_divergence(self, z_mean, z_log_sigma):
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
                self.reconstruct_loss() +
                self.calulate_kl_divergence(z_mean, z_log_sigma),
                name="vae_cost")
            self.cost += l2_reg
