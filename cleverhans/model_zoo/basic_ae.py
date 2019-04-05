"""
A pure TensorFlow implementation of a convolutional neural network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import functools

import tensorflow as tf

from cleverhans import initializers
from cleverhans.model import Model
from PIL import Image

class ModelBasicAE(Model):
  def __init__(self, scope, n_fcc, n_hidden, **kwargs):
    del kwargs
    Model.__init__(self, scope, n_fcc, n_hidden, locals())
    self.n_fcc = n_fcc
    self.n_hidden = n_hidden
    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, shape = (90, 28, 28, 1)))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = tf.shape(x)[3]
    b =np.shape(x)[0]
    h = np.shape(x)[1]
    w = np.shape(x)[2]
    c = np.shape(x)[3]
    #print("shape of x: ", tf.shape(x)[0]," ", tf.shape(x)[1]," ", tf.shape(x)[2]," ", tf.shape(x)[3])
    my_fcc = functools.partial(
        tf.layers.dense, activation=tf.nn.relu)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      #y = my_fcc(tf.reshape(x, [batch_size,h*w*c]), self.n_fcc, name = 'ENC_1')
      y = my_fcc(tf.reshape(x, [batch_size,28*28*1]), self.n_fcc, name = 'ENC_1')
      y = my_fcc(y, self.n_fcc, name = 'ENC_2')
      z = my_fcc(y, self.n_hidden, name='LATENT' )
      d = my_fcc(z, self.n_fcc, name='DEC_1')
      d = my_fcc(d, self.n_fcc, name='DEC_2')
      #recon = my_fcc(d, h*w*c, activation = tf.nn.sigmoid, name='RECON')
      recon = my_fcc(d, 28*28*1, activation = tf.nn.sigmoid, name='RECON')
      #recon = tf.reshape(recon, (b, h, w, c))
      recon = tf.reshape(recon, (batch_size, 28, 28, 1))
      return {
        'LATENT': z,
        'RECON': recon
      }
