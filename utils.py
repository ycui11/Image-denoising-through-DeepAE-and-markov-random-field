
#########################################################################################################################
# Author: Safa Messaoud                                                                                                 #
# E-Mail: messaou2@illinois.edu                                                                                         #
# Instituation: University of Illinois at Urbana-Champaign                                                              #
# Course: ECE 544_na Fall 2017                                                                                          #
# Date: July 2017                                                                                                       #
#                                                                                                                       #
# Description: this script contains helper functions:                                                                   #
#            batch_norm : batch normalization                                                                           #
#            add_noise  : injecting noise into a batch                                                                  #
#########################################################################################################################


import numpy as np
import tensorflow as tf

def batch_norm(x, n_out, phase_train):
  """
  Batch normalization on convolutional maps.
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  Args:
      x:           Tensor, 4D BHWD input maps
      n_out:       integer, depth of input maps
  Return:
      normed:      batch-normalized maps
  """
  
  with tf.variable_scope('bn'):
      beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
      gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
      ema = tf.train.ExponentialMovingAverage(decay=0.5)

      def mean_var_with_update():
          ema_apply_op = ema.apply([batch_mean, batch_var])
          with tf.control_dependencies([ema_apply_op]):
              return tf.identity(batch_mean), tf.identity(batch_var)

      mean, var = tf.cond(phase_train,
                          mean_var_with_update,
                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
      normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

  return normed    


def add_noise(images):
  """
  Create a noisy version of a given batch of image 
  by generating a random mask over pixels with values 
  between 0 and 1. With probability 0.8, keep the pixel's
  original value. Set the pixel's value to 1 with 
  probability 0.1 and to 0 with probability 0.1 as well.

  Inputs:
    images: a batch of images.
  Returns:
    noisy_images: a noisy batch of images.
  """
  noisy_images=images.copy()
  index=np.random.choice(10, size=(noisy_images.shape[0],noisy_images.shape[1]))
  for i in range(index.shape[0]):
      for j in range(index.shape[1]):
          if 1<index[i][j]<3:
              noisy_images[i][j]=0
          if 4<index[i][j]<6:
              noisy_images[i][j]=1
  return noisy_images
