import tensorflow as tf

def crossEntropy(obs, actual, offset=1e-7):
    """Binary cross-entropy, per training example"""
    # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
    with tf.name_scope("cross_entropy"):
        # bound by clipping to avoid nan
        obs_ = tf.clip_by_value(obs, offset, 1 - offset)
        return -tf.reduce_sum(actual * tf.log(obs_) +
                              (1 - actual) * tf.log(1 - obs_), 1)

def l1_loss(obs, actual):
    """L1 loss (a.k.a. LAD), per training example"""
    # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
    with tf.name_scope("l1_loss"):
        return tf.reduce_sum(tf.abs(obs - actual), 1)


def l2_loss(obs, actual):
    """L2 loss (a.k.a. Euclidean / LSE), per training example"""
    # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
    with tf.name_scope("l2_loss"):
        return tf.reduce_sum(tf.square(obs - actual), 1)

def kullbackLeibler(mu, log_sigma):
    """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
    # (tf.Tensor, tf.Tensor) -> tf.Tensor
    with tf.name_scope("KL_divergence"):
        # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
        return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu ** 2 -
                                    tf.exp(2 * log_sigma), 1)
