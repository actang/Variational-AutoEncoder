import tensorflow as tf
from vae.layers import FullyConnectedLayer
from vae.distribution import Distribution
from vae.autoencoder import AutoEncoder


class VariationalAutoEncoder(AutoEncoder):
    """
    Variational Autoencoder
    """
    def __init__(self, input_size, architecture, batch_size=128,
                 distribution=Distribution("normal"), learning_rate=1e-3,
                 l2_reg=1e-5, sesh=None, name="Variational Autoencoder"):
        # Inherit autoencoder
        AutoEncoder.__init__(self, input_size=input_size,
                             architecture=architecture, batch_size=batch_size,
                             learning_rate=learning_rate,  l2_reg=l2_reg,
                             sesh=sesh, name=name)

        # Initialize latent space distribution
        self.z_log_sigma = None
        self.z_sampled = None
        # self.x_reconstructed_ = None
        # self.z_ = None
        self.distribution = distribution

        # for handle in (self.x, self.dropout, self.z_mean,
        #                self.z_log_sigma, self.x_reconstructed, self.z_,
        #                self.x_reconstructed_, self.cost,
        #                self.global_step, self.training_operation):
        #     tf.add_to_collection("vae_parameters", handle)

        # if load_model is not None:
        #     # Load existing model
        #     model_datetime, model_name = os.path.basename(load_model).\
        #         split("_vae_")
        #     self.datetime = "{}_reloaded".format(model_datetime)
        #     *model_architecture, _ = re.split("_|-", model_name)
        #     self.architecture = [int(n) for n in model_architecture]
        #     meta_graph = os.path.abspath(load_model)
        #     tf.train.import_meta_graph(meta_graph + ".meta").\
        #         restore(self.sesh, meta_graph)
        #     (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
        #      self.x_reconstructed, self.z_, self.x_reconstructed_,
        #      self.cost, self.global_step,
        #      self.train_op) = self.sesh.graph.get_collection("vae_parameters")

    def latent_space(self, pre_latent):
        input_size = pre_latent.get_shape().as_list()

        # flatten
        if len(self.architecture) > 1 \
                and self.architecture[-2]['layer'] != "fullyconnected":
            filter_size = input_size[1] * input_size[2] * input_size[3]
            input_size = [input_size[0], filter_size]
            pre_latent = tf.reshape(pre_latent, input_size)

        # Latent variable distribution is a multivariate normal with mean
        # z_mean and diagonal covariance z_log_sigma
        z_mean_layer = FullyConnectedLayer(
            input_size=input_size[-1],
            output_size=self.architecture[-1]['layer_size'],
            scope="fully_connected_layer_encoder_latent_mean",
            dropout=self.architecture[-1]['dropout'],
            activation=self.architecture[-1]['activation']
        )
        z_mean = z_mean_layer(pre_latent)

        z_log_sigma_layer = FullyConnectedLayer(
            input_size=input_size[-1],
            output_size=self.architecture[-1]['layer_size'],
            scope="fully_connected_layer_encoder_latent_sigma",
            dropout=self.architecture[-1]['dropout'],
            activation=self.architecture[-1]['activation']
        )
        z_log_sigma = z_log_sigma_layer(pre_latent)

        output_size = z_mean.get_shape().as_list()
        self.shapes.append((input_size, output_size))

        return z_mean, z_log_sigma

    def _build_graph(self):
        pre_latent = self.encoder()
        self.latent, self.z_log_sigma = self.latent_space(pre_latent)
        self.z_sampled = self.sample_latent_space(self.latent, self.z_log_sigma)
        self.x_reconstructed = tf.identity(self.decoder(self.z_sampled),
                                           name="x_reconstructed")

        # with tf.name_scope("latent"):
        #     self.z_ = tf.placeholder_with_default(
        #         tf.random_normal([1, self.architecture[-1]['layer_size']]),
        #         shape=[None, self.architecture[-1]['layer_size']],
        #         name="latent")
        # self.x_reconstructed_ = tf.identity(self.decoder(self.z_),
        #                                     name="x_reconstructed_")

        self.calculate_cost()
        self.optimization_function()
        self.sesh.run(tf.initialize_all_variables())

    def latent_gradient(self, z, image_grad):
        # compute gradients for manifold exploration
        # z_grad = tf.gradients(self.reconstruct_x(z), [z],
        #                       grad_ys=image_grad)[0]
        # return z_grad
        pass

    def sample_latent_space(self, z_mean, z_log_sigma):
        # Sample the latent distribution. Only one draw is necessary as long as
        # minibatch large enough (>100)
        z = self.distribution.sample_distribution(z_mean, z_log_sigma)
        return z

    def calulate_kl_divergence(self):
        # Kullback-Leibler divergence
        kl_loss = self.distribution.kl_divergence(self.latent, self.z_log_sigma)
        return kl_loss

    def encode(self, x):
        """
        Probabilistic encoder from inputs to latent distribution parameters.
        {inference network q(z|x)}
        """
        feed_dict = {self.x: x}
        return self.sesh.run([self.latent, self.z_log_sigma],
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
            feed_dict.update({self.z_sampled: zs})
        # Else zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sesh.run(self.x_reconstructed, feed_dict=feed_dict)

    def vae(self, x):
        """
        End-to-end autoencoder.
        """
        return self.decode(self.sample_latent_space(*self.encode(x)))

    def calculate_cost(self):
        with tf.name_scope("l2_regularization"):
            regularizers = [
                tf.nn.l2_loss(var) for var in
                self.sesh.graph.get_collection("trainable_variables") if
                "weights" in var.name
            ]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("total_cost"):
            # Average over minibatch
            self.cost = tf.reduce_mean(
                self.reconstruct_loss() + self.calulate_kl_divergence(),
                name="vae_cost"
            )
            self.cost += l2_reg
