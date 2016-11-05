import tensorflow as tf


def cross_entropy(obs, actual, offset=1e-7):
    """
    Calculate binary cross-entropy loss, per training example.
    """
    with tf.name_scope("cross_entropy"):
        obs_ = tf.clip_by_value(obs, offset, 1 - offset)
        return -tf.reduce_sum(actual * tf.log(obs_) +
                              (1 - actual) * tf.log(1 - obs_), 1)

def l1_loss(obs, actual):
    """
    Calculate L1 loss (a.k.a. LAD), per training example.
    """
    with tf.name_scope("l1_loss"):
        return tf.reduce_sum(tf.abs(obs - actual), 1)


def l2_loss(obs, actual):
    """
    Calculate L2 loss (a.k.a. Euclidean / LSE), per training example.
    """
    with tf.name_scope("l2_loss"):
        return tf.reduce_sum(tf.square(obs - actual), 1)


def mse_loss(obs, actual):
    with tf.name_scope("mse_loss"):
        return tf.reduce_sum(tf.square(obs - actual))