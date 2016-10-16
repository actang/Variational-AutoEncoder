from cs294_129.layers import layer_composition, FullyConnectedLayer
from cs294_129.distribution import sample_gaussian, cross_entropy, kl_divergence
from datetime import datetime
import tensorflow as tf
import re
import os


class AutoEncoder:

    def __init__(self,
                 architecture=None,
                 # Currently only support fully connected layer,
                 # in the future architecture can be changed to a list of
                 # tuples, which stores the layer type and neuron amount
                 load_model=None,
                 batch_size=128,
                 learning_rate=1e-3,
                 dropout=1.,
                 lambda_l2_reg=0.,
                 activation=tf.nn.elu,
                 squashing=tf.nn.sigmoid,
                 ):
        """
        (Re)build a symmetric VAE model with given:

        * architecture (list of nodes per encoder layer); e.g.
        [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
        & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]
        """

        # Assign parameters to the object
        self.architecture = architecture
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.lambda_l2_reg = lambda_l2_reg
        self.activation = activation
        self.squashing = squashing

        # Start a Tensorflow session
        self.sesh = tf.Session()

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
            handles = self.sesh.graph.get_collection("to_restore")
        else:
            # Train a new graphical model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            handles = self._build_graph()
            for handle in handles:
                tf.add_to_collection("to_restore", handle)
            self.sesh.run(tf.initialize_all_variables())

        # Unpack handles for tensor ops to feed or fetch
        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.global_step, self.train_op) = handles

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def _build_graph(self):
        x_in = tf.placeholder(tf.float32,
                              shape=[None,  # enables variable batch size
                                     self.architecture[0]], name="x")
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        # encoding / "recognition": q(z|x)
        # hidden layers reversed for function composition: outer -> inner
        encoding = [FullyConnectedLayer(size=hidden_size,
                                        scope="encoding",
                                        dropout=dropout,
                                        activation=self.activation)
                    for hidden_size in reversed(self.architecture[1:-1])]
        h_encoded = layer_composition(encoding)(x_in)

        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = FullyConnectedLayer(size=self.architecture[-1],
                                     scope="z_mean",
                                     dropout=dropout,
                                     activation=tf.identity)(h_encoded)
        z_log_sigma = FullyConnectedLayer(size=self.architecture[-1],
                                          scope="z_log_sigma",
                                          dropout=dropout,
                                          activation=tf.identity)(h_encoded)
        # kingma & welling: only 1 draw necessary as long as minibatch large
        # enough (>100)
        z = sample_gaussian(z_mean, z_log_sigma)

        # decoding / "generative": p(x|z)
        # assumes symmetry
        decoding = [FullyConnectedLayer(size=hidden_size,
                                        scope="decoding",
                                        dropout=dropout,
                                        activation=self.activation)
                    for hidden_size in self.architecture[1:-1]]

        # final reconstruction: restore original dims, squash outputs [0, 1]
        decoding.insert(0,
                        FullyConnectedLayer(
                            size=self.architecture[0],
                            scope="x_decoding",
                            dropout=dropout,
                            activation=self.squashing
                        ))
        x_reconstructed = tf.identity(layer_composition(decoding)(z),
                                      name="x_reconstructed")

        # reconstruction loss: mismatch b/w x & x_reconstructed
        # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis
        rec_loss = cross_entropy(x_reconstructed, x_in)

        # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true
        # posterior
        kl_loss = kl_divergence(z_mean, z_log_sigma)

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in
                            self.sesh.graph.get_collection(
                                "trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # average over minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            cost += l2_reg

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            #  gradient clipping
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar)
                       for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped,
                                                 global_step=global_step,
                                                 name="minimize_cost")

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)
        with tf.name_scope("latent_in"):
            z_ = tf.placeholder_with_default(
                tf.random_normal([1, self.architecture[-1]]),
                shape=[None, self.architecture[-1]],
                name="latent_in")
        x_reconstructed_ = layer_composition(decoding)(z_)

        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed,
                z_, x_reconstructed_, cost, global_step, train_op)


    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean,
                              self.z_log_sigma],
                             feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input
        space; a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            zs = (self.sesh.run(zs) if hasattr(zs, "eval") else zs)
            # coerce to np.array
            feed_dict.update({self.z_: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sesh.run(self.x_reconstructed_,
                             feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(
            sample_gaussian(*self.encode(x))
        )
