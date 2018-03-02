
""" Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

import configuration
from model import Model


slim = tf.contrib.slim
layers = tf.contrib.layers

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

# def _get_init_fn():
#   """Returns a function run by the chief worker to warm-start the training.

#   Note that the init_fn is only run when initializing the model during the very
#   first global step.

#   Returns:
#     An init function run by the supervisor.
#   """
#   if FLAGS.checkpoint_path is None:
#     return None

#   # Warn the user if a checkpoint exists in the train_dir. Then we'll be
#   # ignoring the checkpoint anyway.
#   if tf.train.latest_checkpoint(FLAGS.train_dir):
#     tf.logging.info(
#         'Ignoring --checkpoint_path because a checkpoint already exists in %s'
#         % FLAGS.train_dir)
#     return None

#   exclusions = []
#   if FLAGS.checkpoint_exclude_scopes:
#     exclusions = [scope.strip()
#                   for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

#   # TODO(sguada) variables.filter_variables()
#   variables_to_restore = []
#   for var in slim.get_model_variables():
#     excluded = False
#     for exclusion in exclusions:
#       if var.op.name.startswith(exclusion):
#         excluded = True
#         break
#     if not excluded:
#       variables_to_restore.append(var)

#   if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
#     checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
#   else:
#     checkpoint_path = FLAGS.checkpoint_path

#   tf.logging.info('Fine-tuning from %s' % checkpoint_path)

#   return slim.assign_from_checkpoint_fn(
#       checkpoint_path,
#       variables_to_restore,
#       ignore_missing_vars=FLAGS.ignore_missing_vars)

# # Create the model
# predictions = vgg.vgg_16(images)
#
# train_op = slim.learning.create_train_op(...)
#
# # Specify where the Model, trained on ImageNet, was saved.
# model_path = '/path/to/pre_trained_on_imagenet.checkpoint'
#
# # Specify where the new model will live:
# log_dir = '/path/to/my_pascal_model_dir/'
#
# # Restore only the convolutional layers:
# variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
# init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
#
# # Start training.
# slim.learning.train(train_op, log_dir, init_fn=init_fn)

def main(_):

  with tf.Graph().as_default() as graph:

    assert FLAGS.mode in ['train', 'validation', 'inference']

    # Build the model.
    if FLAGS.mode == "inference":
      model_inference  = Model(mode="inference")
      model_inference.build()
    else:
      model_train = Model(mode="train")
      model_train.build()
      model_validation = Model(mode="validation")
      model_validation.build()

    # train_dir path in each the combination of hyperparameters
    # train_dir = configuration.hyperparameter_dir(FLAGS.train_dir)
    train_dir = FLAGS.train_dir

    # Print all trainable variables
    for var in tf.trainable_variables():
      print(var.name)

    # save the graph structure
    if FLAGS.mode == "inference":
      graph_def = graph.as_graph_def()
      with gfile.GFile(os.path.join(train_dir, FLAGS.output_file), 'wb') as f:
        f.write(graph_def.SerializeToString())
      return

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (FLAGS.num_examples / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.constant(FLAGS.initial_learning_rate)
    def _learning_rate_decay_fn(learning_rate, global_step):
      return tf.train.exponential_decay(
                learning_rate,
                global_step,
                decay_steps=decay_steps,
                decay_rate=FLAGS.learning_rate_decay_factor,
                staircase=True)

    # Create an optimizer that performs gradient descent for Discriminator.
    if FLAGS.optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif FLAGS.optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif FLAGS.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate,
                                         beta1=FLAGS.adam_beta1,
                                         beta2=FLAGS.adam_beta2,
                                         epsilon=FLAGS.adam_epsilon)
    elif FLAGS.optimizer == 'ftrl':
      optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif FLAGS.optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif FLAGS.optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif FLAGS.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
      raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)

    # Set up the training ops.
    train_op = layers.optimize_loss(loss=model_train.loss,
                                    global_step=model_train.global_step,
                                    learning_rate=learning_rate,
                                    optimizer=optimizer,
                                    clip_gradients=5.0,
                                    learning_rate_decay_fn=None,
                                    name='loss')

    # Restore only the convolutional layers:
    variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'])
    if FLAGS.pretrained_file_path == None:
        init_fn = None
    else:
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.pretrained_file_path,
                                                 variables_to_restore,
                                                 ignore_missing_vars=True
                                                 )

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=1000)


    # Build the summary operation
    summary_op = tf.summary.merge_all()


    # Training with tf.train.Supervisor.
    sv = tf.train.Supervisor(logdir=train_dir,
                             summary_op=None,       # Do not run the summary services
                             saver=saver,
                             save_model_secs=0,     # Do not run the save_model services
                             init_fn=init_fn)       # use pre-trained model
    with sv.managed_session() as sess:
      tf.logging.info('Starting Session.')

      # Start the queue runners.
      sv.start_queue_runners(sess=sess)
      tf.logging.info('Starting Queues.')

      # sess.run(tf.global_variables_initializer())

      # Run a model
      pre_epochs = 0.0
      for step in range(FLAGS.max_steps):
        start_time = time.time()
        if sv.should_stop():
          break

        _, _global_step, loss =  sess.run([train_op,
                                          sv.global_step,
                                          model_train.loss])

        epochs = _global_step * FLAGS.batch_size / FLAGS.num_examples
        duration = time.time() - start_time

        # Save the model summaries periodically.
        # if int(pre_epochs) < int(epochs):
        if _global_step % 200 == 0:
          start_time_val = time.time()
          count_top_1 = 0.
          num_iter = int(math.ceil(FLAGS.num_examples_val / FLAGS.batch_size_val))
          for step_val in range(num_iter):
            top_1 = sess.run([model_validation.top_1])
            count_top_1 += np.sum(top_1)
          total_sample_count = num_iter * FLAGS.batch_size_val
          top_1_accuracy = count_top_1 / float(total_sample_count)
          duration_val = time.time() - start_time_val
          examples_per_sec = total_sample_count / float(duration_val)

          print("\nEpochs: %.2f (%.1f examples/sec; %.3f sec/one_validation)\n"
                    % (epochs, examples_per_sec, duration_val))
          print("\nvalidation top_1_accuracy: %.3f \n" % (top_1_accuracy))

          summary = tf.Summary()
          summary.ParseFromString(sess.run(summary_op))
          summary.value.add(tag='top1_accuracy_val', simple_value=top_1_accuracy)
          sv.summary_computed(sess, summary)

        pre_epochs = epochs


        # Print the step, loss, and other information periodically for monitoring.
        if _global_step % 10 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            print("Epochs: %.2f global_step: %d loss: %.4f  (%.1f examples/sec; %.3f sec/batch)"
                      % (epochs, _global_step, loss, examples_per_sec, duration))

        # Save the model summaries periodically.
        if _global_step % 200 == 0:
          summary_str = sess.run(summary_op)
          sv.summary_computed(sess, summary_str)

        # Save the model checkpoint periodically.
        if _global_step % FLAGS.save_steps == 0:
          tf.logging.info('Saving model with step %d to disk.' % _global_step)
          sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

    print('complete training...')
    tf.logging.info('Complete training...')



if __name__ == '__main__':
  tf.app.run()

