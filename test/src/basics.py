import tensorflow as tf
import os

flags = tf.flags
FLAGS = flags.FLAGS

#####################
# Basic Information #
#####################
flags.DEFINE_string(
    "classifier_name",
    "lenet5",
    "name for the classifier"
)
flags.DEFINE_string(
    "vae_name",
    "vae1205",
    "name for the VAE"
)
flags.DEFINE_string(
    "dataset_name",
    "MNIST",
    "name for the dataset"
)

#######################
# Dataset Information #
#######################
flags.DEFINE_integer(
    "image_height",
    28,
    "length of height of the input image"
)
flags.DEFINE_integer(
    "image_width",
    28,
    "length of width of the input image"
)
flags.DEFINE_integer(
    "image_channel",
    1,
    "number of color channels of the input image"
)
flags.DEFINE_integer(
    "image_num_categories",
    10,
    "number of categories of the input images"
)
##########################
# Classifier Information #
##########################
flags.DEFINE_integer(
    "classifier_batch_size",
    50,
    "batch size to train the classifier"
)
flags.DEFINE_integer(
    "classifier_max_epochs",
    50,
    "max epochs to train the classifier"
)
flags.DEFINE_float(
    "classifier_learning_rate",
    1e-2,
    "learning rate to train the classifier"
)

###################
# VAE Information #
###################
flags.DEFINE_integer(
    "vae_hidden_size",
    10,
    "size of the hidden layer of the VAE"
)
flags.DEFINE_integer(
    "vae_batch_size",
    50,
    "batch size to train the VAE"
)
flags.DEFINE_float(
    "vae_learning_rate",
    1e-2,
    "learning rate to train the VAE"
)
flags.DEFINE_integer(
    "vae_max_epochs",
    50,
    "max epochs to train the VAE"
)
###################
# I/O Information #
###################
flags.DEFINE_string(
    "data_path",
    "data",
    "path of the data file"
)
flags.DEFINE_string(
    "model_path",
    "models",
    "path for classifier and VAE to store checkpoint models and logging"
)
flags.DEFINE_string(
    "figures_path",
    "figures",
    "path to store the deliverable figures"
)

######################
# Adjust Information #
######################
flags.DEFINE_integer(
    "adjust_total_iterations",
    1000,
    "total iterations to back propagate gradients to input images"
)
flags.DEFINE_integer(
    "adjust_target_class",
    6,
    "target class to change the input images to"
)
flags.DEFINE_integer(
    "adjust_rate",
    1e-4,
    "adjust rate to apply gradients to the input images"
)

######################
# Unused Information #
######################
flags.DEFINE_integer(
    "max_iterations",
    1e12,
    "max iterations"
)
flags.DEFINE_string(
    "classifier_path",
    "classifier_models",
    "path for classifier to store checkpoint models and logging"
)
flags.DEFINE_string(
    "vae_path",
    "vae_models",
    "path for VAE to store checkpoint models and logging"
)

# Make directory if not exists
for directory in (
    FLAGS.data_path,
    FLAGS.model_path,
    FLAGS.figures_path
):
    try:
        os.mkdir(directory)
    except:
        pass