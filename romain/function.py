import numpy as np
import tensorflow as tf

import os
from six.moves import urllib
import sys
import gzip

import matplotlib.pyplot as plt

# CODE FROM https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/g3doc/tutorials/mnist/input_data.py
# CODE FROM https://jmetzen.github.io/2015-11-27/vae.html
# CODE FROM https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py

# WEIGHT INITIALIZATION

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

# DATA LOADING
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
PIXEL_DEPTH = 255
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_data(N_train, N_valid, N_test):
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    x_valid = train_data[:N_valid]
    y_valid = train_labels[:N_valid]
    x_train = train_data[N_valid:N_valid+N_train]
    y_train = train_labels[N_valid:N_valid+N_train]
    x_test = test_data[:N_test]
    y_test = test_labels[:N_test]
    
    print "Information on dataset"
    print "x_train", x_train.shape
    print "y_train", y_train.shape
    print "x_valid", x_valid.shape
    print "y_valid", y_valid.shape
    print "x_test", x_test.shape
    print "y_test", y_test.shape
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

# SOME PLOTTING

def plot_random_mnist(data, idx=None):
    if idx == None:
        idx = np.random.choice(range(data.shape[0]), size=100)
    #plot a few MNIST examples
    canvas = np.zeros((28*10, 10*28))
    for i in range(10):
        for j in range(10):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = data[idx[(i+1)*(j+1)-1]].reshape((28, 28))
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title('MNIST handwritten digits')
    plt.show()
    
# ACCURACY

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])

    
# PREPARING FOR AUTOENCODER
def initialize_ae_weights(n_hidden_recog_1, n_hidden_recog_2, 
                        n_hidden_gener_1,  n_hidden_gener_2, 
                        n_input, n_z):
    all_weights = dict()
    all_weights['weights_recog'] = {
        'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
        'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
        'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
        'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
    all_weights['biases_recog'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
    all_weights['weights_gener'] = {
        'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
        'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
        'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
        'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
    all_weights['biases_gener'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
    return all_weights

def recognition_network(x, weights, biases, transfer_fct):
    # Generate probabilistic encoder (recognition network), which
    # maps inputs onto a normal distribution in latent space.
    # The transformation is parametrized and can be learned.
    layer_1 = transfer_fct(tf.add(tf.matmul(x, weights['h1']), 
                                       biases['b1'])) 
    layer_2 = transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                       biases['b2'])) 
    z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                    biases['out_mean'])
    z_log_sigma_sq = \
        tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
               biases['out_log_sigma'])
    return (z_mean, z_log_sigma_sq)

def generator_network(z, weights, biases, transfer_fct):
    # Generate probabilistic decoder (decoder network), which
    # maps points in latent space onto a Bernoulli distribution in data space.
    # The transformation is parametrized and can be learned.
    layer_1 = transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                       biases['b1'])) 
    layer_2 = transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                       biases['b2'])) 
    x_reconstr_mean = \
        tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                             biases['out_mean']))
    return x_reconstr_mean

def create_autoencoder(x, network_architecture, transfer_fct):
    # Initialize autoencode network weights and biases
    network_weights = initialize_ae_weights(**network_architecture)

    # Use recognition network to determine mean and 
    # (log) variance of Gaussian distribution in latent
    # space
    z_mean, z_log_sigma_sq = \
        recognition_network(x, network_weights["weights_recog"], 
                                  network_weights["biases_recog"], transfer_fct)

    # Draw one sample z from Gaussian distribution
    n_z = network_architecture["n_z"]
    eps = tf.random_normal(tf.pack([tf.shape(x)[0], n_z]), 0, 1, dtype = tf.float32)

    # z = mu + sigma*epsilon
    z = tf.add(z_mean, 
                    tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

    # Use generator to determine mean of
    # Bernoulli distribution of reconstructed input
    x_reconstr_mean = \
        generator_network(z, network_weights["weights_gener"],
                                network_weights["biases_gener"], transfer_fct)
        
    # create a gradient to perform manifold exploration
    x_grad = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS))
    z_grad = tf.gradients(x_reconstr_mean, [z], grad_ys=x_grad)[0]
    
    return z, x_reconstr_mean, z_mean, z_log_sigma_sq, x_grad, z_grad

def create_encoder_loss_optimizer(x, x_hat, z_mean, z_log_sigma_sq, learning_rate):
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution 
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluatio of log(0.0)
    reconstr_loss = \
        -tf.reduce_sum(x * tf.log(1e-10 + x_hat)
                       + (1-x) * tf.log(1e-10 + 1 - x_hat),
                       1)
    reconstr_error = tf.reduce_sum(reconstr_loss)
    # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
    ##    between the distribution in latent space induced by the encoder on 
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq 
                                       - tf.square(z_mean) 
                                       - tf.exp(z_log_sigma_sq), 1)
    cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
    # Use ADAM optimizer
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return reconstr_error, cost, optimizer 


# TRAINING THE RECOGNITION MODEL
def create_recognition_network(y, n_z):
    z_input = tf.placeholder(tf.float32, shape=(None, n_z))
    fc_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))
    fc_weights = tf.Variable(tf.truncated_normal([n_z, NUM_LABELS],
                                            stddev=0.1,
                                            dtype=tf.float32))
    scores = tf.matmul(z_input, fc_weights) + fc_biases
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(scores, y))
    return z_input, scores, loss

def create_recognition_loss_optimizer(scores_reco, loss_reco, learning_rate):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss_reco)



# NOW THE LECUN
def create_classifier_network(learning_rate_lenet):
    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    PIXEL_DEPTH = 255
    NUM_LABELS = 10
    SEED = 66478  # Set to None for random seed.
    BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 64
    EVAL_FREQUENCY = 100 # Number of steps between evaluations.
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
      tf.float32,
      shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(None,))

    eval_data = tf.placeholder(
      tf.float32,
      shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    eval_labels = tf.placeholder(tf.int64, shape=(None,))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.1,
      seed=SEED, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=tf.float32))
    
    def model_lenet(data, train=False):
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        data_tran = data - 0.5
        conv = tf.nn.conv2d(data_tran,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
              hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        scores = tf.matmul(hidden, fc2_weights) + fc2_biases
        return scores
    

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.

    logits = model_lenet(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))
    optimizer = tf.train.AdamOptimizer(learning_rate_lenet).minimize(loss)
    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_score = model_lenet(eval_data)
    eval_prediction = tf.nn.softmax(eval_score)
    eval_saliency_grad = tf.gradients(eval_score, [eval_data], grad_ys=[tf.one_hot(eval_labels, NUM_LABELS)])[0]
    return train_data_node, train_labels_node, logits, loss, optimizer, \
        eval_data, eval_labels, eval_score, eval_prediction, eval_saliency_grad, train_prediction
    
