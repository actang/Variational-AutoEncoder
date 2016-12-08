from __future__ import absolute_import, division, print_function
import os
import prettytensor as pt
import tensorflow as tf
from deconv import deconv2d
from tqdm import trange
from plot import dataset_sample_plot, vae_plot_sampled, vae_plot_reconstructed, \
    plot_all_images
from basics import FLAGS
import numpy as np

logging = tf.logging
meta_model_name = "{0}_{1}_{2}_epochs_{3}_{4}_epochs.ckpt".format(
    FLAGS.dataset_name,
    FLAGS.vae_name,
    FLAGS.vae_max_epochs,
    FLAGS.classifier_name,
    FLAGS.classifier_max_epochs
)

meta_model_path = os.path.join(
    FLAGS.model_path,
    meta_model_name
)

##################
# Import Dataset #
##################
from tensorflow.examples.tutorials.mnist import input_data
data_directory = os.path.join(FLAGS.data_path, FLAGS.dataset_name)
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
dataset = input_data.read_data_sets(
    data_directory,
    reshape=False,
    one_hot=True
)


######################
# Random Sample Plot #
######################
dataset_sample_plot(FLAGS, dataset)


#################
# Construct VAE #
#################

def naive_encoder(images):
    with tf.variable_scope("vae"):
        images = pt.wrap(images)
        return (
            images
            .conv2d(5, 32, stride=2)
            .conv2d(5, 64, stride=2)
            .conv2d(5, 128, edges='VALID')
            .dropout(keep_prob=0.9)
            .flatten()
            .fully_connected(FLAGS.vae_hidden_size * 2, activation_fn=None)
        ).tensor


def naive_decoder(z=None):
    with tf.variable_scope("vae"):
        epsilon = tf.random_normal([
            FLAGS.vae_batch_size,
            FLAGS.vae_hidden_size
        ])
        if z is None:
            mean = None
            stddev = None
            input_sample = epsilon
        else:
            mean = z[:, :FLAGS.vae_hidden_size]
            stddev = tf.sqrt(tf.exp(z[:, FLAGS.vae_hidden_size:]))
            input_sample = mean + epsilon * stddev
        return (
                   pt.wrap(input_sample)
                   .reshape([FLAGS.vae_batch_size, 1, 1, FLAGS.vae_hidden_size])
                   .deconv2d(3, 128, edges='VALID')
                   .deconv2d(5, 64, edges='VALID')
                   .deconv2d(5, 32, stride=2)
                   .deconv2d(5, FLAGS.image_channel, stride=2,
                             activation_fn=tf.nn.sigmoid)
               ).tensor, mean, stddev


def get_vae_cost(mean, stddev, epsilon=1e-8):
    return tf.reduce_sum(
        0.5 * (
            tf.square(mean) + tf.square(stddev) -
            2.0 * tf.log(stddev + epsilon) - 1.0
        )
    )


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    output_tensor = tf.reshape(output_tensor, [-1])
    target_tensor = tf.reshape(target_tensor, [-1])
    return tf.reduce_sum(
        - target_tensor * tf.log(output_tensor + epsilon) -
        (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon)
    )

with tf.name_scope("vae"):

    vae_images_placeholder = tf.placeholder(
        tf.float32,
        [
            FLAGS.vae_batch_size,
            FLAGS.image_height,
            FLAGS.image_width,
            FLAGS.image_channel
        ]
    )

    with pt.defaults_scope(
            activation_fn=tf.nn.elu,
            batch_normalize=True,
            learned_moments_update_rate=0.0003,
            variance_epsilon=0.001,
            scale_after_normalization=True
    ):
        # Use naive autoencoder
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("vae_model"):
                vae_z = naive_encoder(vae_images_placeholder)
                vae_x_, vae_z_mean, vae_z_std = naive_decoder(vae_z)
        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("vae_model", reuse=True):
                vae_sampled_x_ = naive_decoder()[0]

    vae_distribution_loss = get_vae_cost(
        vae_z_mean,
        vae_z_std
    )

    var_recontruction_loss = get_reconstruction_cost(
        vae_x_,
        vae_images_placeholder
    )

    vae_loss = pt.create_composite_loss(
        losses=[
            vae_distribution_loss,
            var_recontruction_loss
        ]
    )

    vae_x_grad = tf.placeholder(
        tf.float32,
        [
            FLAGS.vae_batch_size,
            FLAGS.image_height,
            FLAGS.image_width,
            FLAGS.image_channel
        ]
    )

    vae_z_grad = tf.gradients(
        ys=vae_x_,
        xs=[vae_z],
        grad_ys=vae_x_grad
    )[0]

    vae_optimizer = tf.train.AdamOptimizer(
        FLAGS.vae_learning_rate,
        epsilon=1.0
    ).minimize(
        loss=vae_loss,
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="vae"
        )
    )


########################
# Construct Classifier #
########################

def construct_lenet(lenet_images, lenet_labels):
    with tf.variable_scope("classifier"):
        images = pt.wrap(lenet_images)
        with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
            return (
                images
                .conv2d(5, 20)
                .max_pool(2, 2)
                .conv2d(5, 50)
                .max_pool(2, 2)
                .flatten()
                .fully_connected(500)
                .softmax_classifier(FLAGS.image_num_categories,
                                    lenet_labels)
            )

with tf.name_scope("classifier"):
    classifier_images_placeholder = tf.placeholder(
        tf.float32,
        [
            FLAGS.classifier_batch_size,
            FLAGS.image_height,
            FLAGS.image_width,
            FLAGS.image_channel
        ]
    )
    classifier_labels_placeholder = tf.placeholder(
        tf.float32,
        [
            FLAGS.classifier_batch_size,
            FLAGS.image_num_categories
        ]
    )

    classifier_result = construct_lenet(
        classifier_images_placeholder,
        classifier_labels_placeholder
    )
    classifier_grads = tf.gradients(
        ys=classifier_result.loss,
        xs=[classifier_images_placeholder]
    )[0]
    classifier_accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(classifier_result.softmax, 1),
                tf.argmax(classifier_labels_placeholder, 1)
            ),
            tf.float32
        )
    )

    classifier_optimizer = tf.train.AdamOptimizer(
        FLAGS.classifier_learning_rate
    ).minimize(
        loss=classifier_result.loss,
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="classifier"
        )
    )

meta_model_exist_flag = False
for f in os.listdir(FLAGS.model_path):
    if meta_model_name in f:
        meta_model_exist_flag = True

if not meta_model_exist_flag:
    init = tf.initialize_all_variables()
    classifier_train_updates_per_epoch = \
        dataset.train.num_examples // FLAGS.classifier_batch_size
    classifier_test_updates_per_epoch = \
        dataset.test.num_examples // FLAGS.classifier_batch_size
    vae_train_updates_per_epoch = \
        dataset.train.num_examples // FLAGS.vae_batch_size

    with tf.Session() as sess:
        sess.run(init)
        ####################
        # Train Classifier #
        ####################
        for epoch in trange(
                FLAGS.classifier_max_epochs,
                desc="{0}: Number of Epochs".format(FLAGS.classifier_name),
                leave=False
        ):
            classifier_training_loss = 0.0
            for i in trange(
                    classifier_train_updates_per_epoch,
                    desc="Training in Epoch {0}".format(epoch + 1),
                    leave=False
            ):
                training_images, training_labels = dataset.train.next_batch(
                    FLAGS.classifier_batch_size
                )
                _, classifier_runtime_loss = sess.run(
                    fetches=[
                        classifier_optimizer,
                        classifier_result.loss
                    ],
                    feed_dict={
                        classifier_images_placeholder: training_images,
                        classifier_labels_placeholder: training_labels
                    }
                )
                classifier_training_loss += classifier_runtime_loss
            classifier_training_loss /= dataset.train.num_examples
            print(
                "\n Training loss after %d epoch %f" % (
                    epoch + 1,
                    classifier_training_loss
                )
            )

            classifier_testing_accuracy = 0.0
            for i in trange(
                    classifier_test_updates_per_epoch,
                    desc="Validation in Epoch {0}".format(epoch + 1),
                    leave=False
            ):
                testing_images, testing_labels = dataset.validation.next_batch(
                    FLAGS.classifier_batch_size
                )
                classifier_runtime_accuracy = sess.run(
                    fetches=[
                        classifier_accuracy
                    ],
                    feed_dict={
                        classifier_images_placeholder: testing_images,
                        classifier_labels_placeholder: testing_labels
                    }
                )
                classifier_testing_accuracy += classifier_runtime_accuracy[0]
            classifier_testing_accuracy /= classifier_test_updates_per_epoch
            print(
                "\n Validation accuracy after %d epoch %g%%" % (
                    epoch + 1,
                    classifier_testing_accuracy * 100
                )
            )

        #############
        # Train VAE #
        #############
        for epoch in trange(
                FLAGS.vae_max_epochs,
                desc="{0}: Number of Epochs".format(FLAGS.vae_name),
                leave=False
        ):
            vae_training_loss = 0.0
            for i in trange(
                    vae_train_updates_per_epoch,
                    desc="Training in Epoch {0}".format(epoch + 1),
                    leave=False
            ):
                training_images, _ = dataset.train.next_batch(
                    FLAGS.vae_batch_size
                )
                _, vae_runtime_loss = sess.run(
                    fetches=[
                        vae_optimizer,
                        vae_loss
                    ],
                    feed_dict={
                        vae_images_placeholder: training_images
                    }
                )
                vae_training_loss += vae_runtime_loss
            vae_training_loss /= dataset.train.num_examples
            print(
                "\n Training loss after %d epoch %f" % (
                    epoch + 1,
                    vae_training_loss
                )
            )

            # Output sampled images
            vae_sampled_images = sess.run(vae_sampled_x_)
            vae_plot_sampled(
                FLAGS,
                vae_sampled_images,
                epoch
            )

            # Output reconstructed images
            testing_images, _ = dataset.test.next_batch(
                FLAGS.vae_batch_size
            )
            vae_reconstructed_images = sess.run(
                fetches=[
                    vae_x_,
                ],
                feed_dict={
                    vae_images_placeholder: testing_images
                }
            )[0]
            vae_plot_reconstructed(
                FLAGS,
                testing_images,
                vae_reconstructed_images,
                epoch
            )

        save_path = tf.train.Saver().save(
            sess,
            meta_model_path
        )


training_images, true_labels = dataset.train.next_batch(
    FLAGS.classifier_batch_size
)

false_labels = np.zeros_like(true_labels)
for i in range(false_labels.shape[0]):
    false_labels[i][FLAGS.adjust_target_class - 1] = 1

##################
# Fooling Images #
##################
# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     saver.restore(sess, meta_model_path)
#     fooling_images_training_images = training_images.copy()
#     for t in trange(
#             FLAGS.adjust_total_iterations,
#             desc="Number of Iterations to Generate Fooling Images",
#             leave=False
#     ):
#         fooling_image_grads, fooling_image_scores = sess.run(
#             fetches=[
#                 classifier_grads,
#                 classifier_result.softmax
#             ],
#             feed_dict={
#                 classifier_images_placeholder: fooling_images_training_images,
#                 classifier_labels_placeholder: false_labels
#             }
#         )
#
#         if t % 100 == 0:
#             plot_all_images(FLAGS, fooling_images_training_images, t,
#                             theme="fooling", title="reconstructed")
#             plot_all_images(FLAGS, fooling_image_grads, t,
#                             theme="fooling", title="gradients")
#
#         fooling_images_training_images += FLAGS.adjust_rate * \
#                                           fooling_image_grads
#

#####################
# Image Transformer #
#####################
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, meta_model_path)
    transformer_training_images = training_images.copy()
    # compute local z as a starting point
    transformer_z_loc = sess.run(
        fetches=[vae_z],
        feed_dict={
            vae_images_placeholder: transformer_training_images
        }
    )[0]
    for t in trange(
            FLAGS.adjust_total_iterations,
            desc="Number of Iterations to Generate Transform Images",
            leave=False
    ):
        transformer_reconstructed_images = sess.run(
            fetches=[
                vae_x_,
            ],
            feed_dict={
                vae_z: transformer_z_loc,
            }
        )[0]
        transformer_image_grads = sess.run(
            fetches=[
                classifier_grads,
            ],
            feed_dict={
                classifier_images_placeholder: transformer_reconstructed_images,
                classifier_labels_placeholder: false_labels
            }
        )[0]
        transformer_z_grads = sess.run(
            fetches=[
                vae_z_grad
            ],
            feed_dict={
                vae_x_grad: transformer_image_grads,
                vae_x_: transformer_reconstructed_images,
                vae_z: transformer_z_loc
            }
        )[0]

        if t % 100 == 0:
            plot_all_images(FLAGS, transformer_reconstructed_images, t,
                            theme="transformer", title="reconstructed")

        transformer_z_loc += FLAGS.adjust_rate * transformer_z_grads
