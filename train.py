
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
                                    learning_rate_decay_fn=_learning_rate_decay_fn,
                                    name='loss')

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=1000)


    # Build the summary operation
    summary_op = tf.summary.merge_all()


    # Training with tf.train.Supervisor.
    sv = tf.train.Supervisor(logdir=train_dir,
                             summary_op=None,       # Do not run the summary services
                             saver=saver,
                             save_model_secs=0,     # Do not run the save_model services
                             init_fn=None)          # Not use pre-trained model
    with sv.managed_session() as sess:
      tf.logging.info('Starting Session.')

      # Start the queue runners.
      sv.start_queue_runners(sess=sess)
      tf.logging.info('Starting Queues.')

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
        if _global_step % 100 == 0:
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
        #if _global_step % 10 == 0:
        examples_per_sec = FLAGS.batch_size / float(duration)
        print("Epochs: %.2f global_step: %d loss: %.4f  (%.1f examples/sec; %.3f sec/batch)"
                  % (epochs, _global_step, loss, examples_per_sec, duration))

        # Save the model summaries periodically.
        if _global_step % 100 == 0:
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

