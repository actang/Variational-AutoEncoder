import tensorflow as tf


class Distribution:
    def __init__(self, distribution="normal"):
        self.distribution = distribution

    def sample_distribution(self, mu, log_sigma):
        if self.distribution == "normal":
            return self.sample_gaussian(mu, log_sigma)
        else:
            return None

    def sample_gaussian(self, mu, log_sigma):
        """
        Draw sample from Gaussian with given shape, subject to random noise
        epsilon using reparameterization trick.
        """
        with tf.name_scope("sample_gaussian"):
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma)

    def kl_divergence(self, mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu ** 2 -
                                        tf.exp(2 * log_sigma), 1)
