#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:03:07 2017

@author: chang
"""
import tensorflow as tf
import configuration
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

FLAGS = tf.app.flags.FLAGS

MODEL_NAME = 'flower'
input_graph_path = FLAGS.infer_flower
checkpoint_path = FLAGS.model_flower

#if not os.path.exists("./log"):
#  print('makedirs ./log')
#  os.makedirs("./log")

# Freezing the Graph
input_saver_def_path = ""
input_binary = True
output_node_names = "InceptionV3/Predictions/Reshape_1"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = FLAGS.log_dir + '/frozen_'+ MODEL_NAME +'.pb'
output_optimized_graph_name = FLAGS.log_dir + '/optimized_'+ MODEL_NAME +'.pb'
clear_devices = True

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")

