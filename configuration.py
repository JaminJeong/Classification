""" Configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


##################
# Training Flags #
##################
tf.app.flags.DEFINE_string('train_dir',
                           None,
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('max_steps',
                            20000,
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('save_steps',
                            1000,
                            'The step per saving model.')
tf.app.flags.DEFINE_string('output_file',
                           'checkpoint.pb',
                           'The name of model.pb file.')

#################
# Dataset Flags #
#################
tf.app.flags.DEFINE_string('dataset_dir',
                           None,
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('batch_size',
                            32,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('batch_size_val',
                            32,
                            'The number of samples for validation in each batch.')
tf.app.flags.DEFINE_integer('num_classes',
                            60,
                            'The number of classes + background label.')
tf.app.flags.DEFINE_integer('num_examples',
                            9537,
                            'The number of samples in total train dataset.')
tf.app.flags.DEFINE_integer('num_examples_val',
                            9537,
                            'The number of samples in total validation dataset.')

#################
# Dataset Flags #
#################
tf.app.flags.DEFINE_integer('image_size', 224,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 8,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


########################
# Learning rate policy #
########################
tf.app.flags.DEFINE_float('initial_learning_rate',
                          0.001,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay',
                          4,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor',
                          0.1,
                          'Learning rate decay factor.')

######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_string('optimizer',
                           'adam',
                           'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                           '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('adam_beta1',
                          0.9,
                          'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2',
                          0.999,
                          'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('adam_epsilon',
                          1e-08,
                          'A small constant for numerical stability.')

####################
# Evaluating Flags #
####################
tf.app.flags.DEFINE_string('eval_dir',
                           'eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir',
                           None,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('checkpoint_step',
                            -1,
                            'The step you want to read model checkpoints.'
                            '-1 means the latest model.')
tf.app.flags.DEFINE_boolean('is_all_checkpoints',
                            False,
                            'Evaluate whether all checkpoints or a specific checkpoint.')
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation', 'test' or 'train'.""")

tf.app.flags.DEFINE_string('pretrained_file_path',
                           None,
                           """pretrained model file for fine tuning""")

###############################
# Frequency of the eval Flags #
###############################
tf.app.flags.DEFINE_integer('eval_interval_secs',
                            20,
                            """How often to run the eval.""")

###################
# Inference Flags #
###################
tf.app.flags.DEFINE_string('mode',
                           '',
                           'what action do you want to infer?')
tf.app.flags.DEFINE_string('frozen_graph',
                           'frozen_graph.pb',
                           'The frozen graph trained.')
tf.app.flags.DEFINE_string('input_file',
                           '',
                           'The demo video file')
tf.app.flags.DEFINE_boolean('is_cam',
                            True,
                            'use webcam or not')


tf.app.flags.DEFINE_string('train_directory', '/tmp/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('test_directory', '/tmp/',
                           'Test data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')
tf.app.flags.DEFINE_string('trash_directory', '/tmp/',
                           'Trash data directory')

tf.app.flags.DEFINE_integer('train_shards', 32,
                            'Number of shards in training tfrecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 2,
                            'Number of shards in validation tfrecord files.')
tf.app.flags.DEFINE_integer('test_shards', 2,
                            'Number of shards in test tfrecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')


tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')


FLAGS = tf.app.flags.FLAGS


# def hyperparameter_dir(input_dir):
#   hp_dir = os.path.join(input_dir, 'model')
#   hp_dir = os.path.join(hp_dir, FLAGS.kernel_depth)
#   hp_dir = os.path.join(hp_dir, FLAGS.optimizer)
#   if FLAGS.optimizer == 'adam':
#     hp_dir = os.path.join(hp_dir, '%.1E' % FLAGS.adam_beta1)
#     hp_dir = os.path.join(hp_dir, '%.2E' % FLAGS.adam_beta2)
#     hp_dir = os.path.join(hp_dir, '%.1E' % FLAGS.adam_epsilon)
#   hp_dir = os.path.join(hp_dir, '%.1E' % FLAGS.initial_learning_rate)
#   hp_dir = os.path.join(hp_dir, '%03d' % FLAGS.num_epochs_per_decay)
#   hp_dir = os.path.join(hp_dir, '%.2E' % FLAGS.learning_rate_decay_factor)
#   print(hp_dir)
#
#   return hp_dir
#
