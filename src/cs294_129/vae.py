from datetime import datetime
import os
import sys

import numpy as np
import tensorflow as tf

from cs294_129.autoencoder import AutoEncoder
from cs294_129.distribution import sample_gaussian
from cs294_129 import plot


class VariationalAutoEncoder:
    """
    Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu,
        "squashing": tf.nn.sigmoid
    }
    RESTORE_KEY = "to_restore"

    def __init__(self,
                 architecture=None,
                 # Currently only support fully connected layer,
                 # in the future architecture can be changed to a list of
                 # tuples, which stores the layer type and neuron amount
                 load_model=None,
                 save_model=True,
                 log_dir="./log",
                 batch_size=128,
                 learning_rate=1e-3,
                 dropout=1.,
                 lambda_l2_reg=0.,
                 activation=tf.nn.elu,
                 squashing=tf.nn.sigmoid):

        self.autoencoder = AutoEncoder(architecture, load_model,
                                       batch_size, learning_rate,
                                       dropout, lambda_l2_reg, activation,
                                       squashing)
        self.sesh = self.autoencoder.sesh

        if save_model:  # Tensorboard
            self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)

    @property
    def step(self):
        """Train step"""
        return self.autoencoder.global_step.eval(session=self.sesh)

    def train(self, X,
              max_iter=np.inf, max_epochs=np.inf,
              cross_validate=True, verbose=True,
              saver=True,
              outdir="./models", plots_outdir="./plots",
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
                x, _ = X.train.next_batch(self.autoencoder.batch_size)
                feed_dict = {self.autoencoder.x_in: x,
                             self.autoencoder.dropout_:
                                 self.autoencoder.dropout}
                fetches = [self.autoencoder.x_reconstructed,
                           self.autoencoder.cost,
                           self.autoencoder.global_step,
                           self.autoencoder.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)

                err_train += cost

                if plot_latent_over_time:
                    while int(round(BASE**pow_)) == i:
                        plot.exploreLatent(self.autoencoder, nx=30, ny=30,
                                           ppf=True, outdir=plots_outdir,
                                           name="explore_ppf30_{}".format(pow_))
                        names = ("train", "validation", "test")

                        datasets = (X.train, X.validation, X.test)
                        for name, dataset in zip(names, datasets):
                            plot.plotInLatent(self.autoencoder, dataset.images,
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
                    plot.plotSubset(self.autoencoder, x, x_reconstructed, n=10,
                                    name="train", outdir=plots_outdir)

                    if cross_validate:
                        x, _ = X.validation.next_batch(
                            self.autoencoder.batch_size)
                        feed_dict = {self.autoencoder.x_in: x}
                        fetches = [self.autoencoder.x_reconstructed,
                                   self.autoencoder.cost]
                        x_reconstructed, cost = self.sesh.run(fetches,
                                                              feed_dict)

                        print("round {} --> CV cost: ".format(i), cost)
                        plot.plotSubset(self.autoencoder, x, x_reconstructed,
                                        n=10, name="cv", outdir=plots_outdir)

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, X.train.epochs_completed, err_train / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    if saver is not None:
                        outfile = os.path.join(
                            os.path.abspath(outdir),
                            "{}_vae_{}".format(self.autoencoder.datetime,
                                               "_".join(map(str,
                                                        self.autoencoder.
                                                            architecture)
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
                i, X.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)
