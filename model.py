from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import image_processing
from vgg import vgg_16

slim = tf.contrib.slim
layers = tf.layers
arg_scope = tf.contrib.framework.arg_scope

FLAGS = tf.app.flags.FLAGS

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def conv2d(inputs,
           filters,
           kernel_size=[3, 3], # [depth, height, width]
           strides=[1, 1],     # [depth, height, width]
           padding='SAME',
           activation=tf.nn.relu,
           weight_decay=0.0005,
           scope=None):

  with tf.variable_scope(scope, 'conv2d'):
    layer = layers.conv2d(inputs=inputs,
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          activation=activation,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    return layer


def max_pooling2d(inputs,
                  pool_size=[2, 2],  # [depth, height, width]
                  strides=[2, 2],    # [depth, height, width]
                  padding='VALID',
                  scope=None):

  with tf.variable_scope(scope, 'max_pooling2d'):
    layer = layers.max_pooling2d(inputs=inputs,
                                 pool_size=pool_size,
                                 strides=strides,
                                 padding=padding)
    return layer


def average_pooling2d(inputs,
                      pool_size=[7, 7],  # [depth, height, width]
                      strides=[1, 1],    # [depth, height, width]
                      padding='VALID',
                      scope=None):

  with tf.variable_scope(scope, 'avg_pooling2d'):
    layer = layers.average_pooling2d(inputs=inputs,
                                     pool_size=pool_size,
                                     strides=strides,
                                     padding=padding)
    return layer


def dense(inputs,
          num_outputs,
          activation=tf.nn.relu,
          weight_decay=0.0005,
          scope=None):

  with tf.variable_scope(scope, 'fully_connected'):
    layer = layers.dense(inputs=inputs,
                         units=num_outputs,
                         activation=activation,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    return layer


def dropout(inputs,
            rate=0.5,
            training=False,
            scope=None):

  with tf.variable_scope(scope, 'dropout'):
    layer = layers.dropout(inputs=inputs,
                           rate=rate,
                           training=training)
    return layer





class Model(object):
  """
  Author = {Jakub Sochor and Jakub Špaňhel and Adam Herout},
  Title = {BoxCars: Improving Vehicle Fine-Grained Recognition using 3D Bounding Boxes in Traffic Surveillance},
  Year = {2017},
  Eprint = {arXiv:1703.00686},
  """

  def __init__(self, mode):
    """Basic setup.

      Args:
        mode: "train", "eval" or "inference"
    """
    assert mode in ["train", "eval", "validation", "inference"]
    self.mode = mode

    # hyper-parameters for dataset 
    self.image_size = FLAGS.image_size
    self.batch_size = FLAGS.batch_size
    self.batch_size_val = FLAGS.batch_size_val
    if mode == "inference":
        self.batch_size = 1
    self.num_classes = FLAGS.num_classes
    self.num_preprocess_threads = FLAGS.num_preprocess_threads

    # losses
    self.loss = None

    # Global step Tensor.
    self.global_step = None

    print('The mode is %s.' % self.mode)
    print('complete initializing model.')


  def is_training(self):
    """Return true if the model is built for training mode."""
    return self.mode == "train"


  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      inputs: images with 4-D Tensor [batch_size, height, width, channels]
      labels: labels in each angle class
    """
#   if self.mode == "inference":
#     # In inference mode, images are fed via placeholder.
#     with tf.variable_scope('images'):
#       self.images = tf.placeholder(dtype=tf.float32,
#         shape=[None, self.num_frames, self.image_size, self.image_size, 3])

    if self.mode == 'train':
      with tf.variable_scope('images_and_labels'):
        self.images, self.labels = image_processing.distorted_inputs(
                                       batch_size=self.batch_size,
                                       num_preprocess_threads=self.num_preprocess_threads)
        #   self.images = tf.random_normal([self.batch_size, self.image_size, self.image_size, 3], dtype=tf.float32)
        #   self.labels = tf.random_uniform(shape=[self.batch_size, self.num_classes], maxval=2, dtype=tf.int32)

    elif self.mode == 'validation':
      with tf.variable_scope('images_and_labels'):
        self.images, self.labels = image_processing.inputs(
                                          batch_size=self.batch_size_val,
                                          num_preprocess_threads=self.num_preprocess_threads)
        # self.images = tf.random_normal([self.batch_size, self.image_size, self.image_size, 3], dtype=tf.float32)
        # self.labels = tf.random_uniform(shape=[self.batch_size, self.num_classes], maxval=2, dtype=tf.int32)

    else:
      with tf.variable_scope('images_and_labels'):
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[1, FLAGS.image_size, FLAGS.image_size, 3])

    print('complete build inputs.')


  def build_network(self):
    """Builds the image model subgraph and generates image feature maps.
    """
    # dimension convention: depth x height x width x channels
    inputs = self.images

    with tf.variable_scope('model', [inputs]) as sc:
      if self.mode == "validation":
        sc.reuse_variables()

      net, end_points = vgg_16(inputs, num_classes=self.num_classes, is_training=self.is_training())
      print("self.num_classes")
      print(self.num_classes)


    print('complete network build.')

    return net


  def build_model(self):
    """Build the model.
    """
    self.logits = self.build_network()

    if self.mode == "train":
      print("self.labels.shape")
      print(self.labels.shape)

      loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels,
                                                  logits=self.logits,
                                                  scope='loss')
      self.loss = loss

      for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    elif self.mode == "validation":
      self.top_1 = tf.nn.in_top_k(predictions=self.logits,
                                  targets=tf.argmax(self.labels, axis=1), k=1)

    else:
      self.probability = tf.argmax(tf.nn.softmax(self.logits, name="softmax"))

    print('complete model build.')


  def setup_global_step(self):
    """Sets up the global step Tensor."""
    if self.mode == "train":
      self.global_step = tf.Variable(
                            initial_value=0,
                            name="global_step",
                            trainable=False,
                            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

      print('complete setup global_step.')


  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_model()
    self.setup_global_step()

    print('complete model build.\n')

