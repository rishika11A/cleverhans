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

class ModelCls(Model):
  def __init__(self, scope, **kwargs):
    del kwargs
    Model.__init__(self, scope, locals())
    self.n_units = 100
    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, shape = (128, 28,28, 1)))
    #self.fprop(tf.placeholder(tf.float32, shape = (128, 100, 1)))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = tf.shape(x)[3]
    print("channels: ", channels)
    #input_size = 28*28
    #print("shape of x: ", tf.shape(x)[0]," ", tf.shape(x)[1]," ", tf.shape(x)[2]," ", tf.shape(x)[3])
    my_fcc = functools.partial(
        tf.layers.dense, activation=tf.nn.relu)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      y = my_fcc(tf.reshape(x, [batch_size,28*28*1]), self.n_units, name = 'FCC1')
      #y = my_fcc(tf.reshape(z, [batch_size,100]), self.n_units, name = 'FCC1')
      y = my_fcc(y, self.n_units, name = 'FCC2')
      y = my_fcc(y, self.n_units, name = 'FCC3')
      logits = my_fcc(y, 10, activation = tf.nn.sigmoid, name='LOGITS')
      '''
      pred_label = [tf.argmax(logits[i]) for i in range(0,128)]
      print("shape of pred_label: ", np.shape(pred_label))
      pred = np.zeros((128,10))
      sess = tf.Session()
      with sess.as_default():
        for i in range(0, 128):
          pred[i,pred_label[i].eval()] = 1

        pred1 = tf.convert_to_tensor(pred)
        '''
      return {
        self.O_LOGITS: logits,
        'LOGITS': logits,
      }
