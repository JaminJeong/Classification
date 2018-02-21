""" Configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


##################
# Training Flags #
##################

tf.app.flags.DEFINE_string('log_dir', '', 'log directory.')
tf.app.flags.DEFINE_string('infer_flower', "", 'infer repetition pb file.')
tf.app.flags.DEFINE_string('model_flower', "", 'infer repetition model file')

