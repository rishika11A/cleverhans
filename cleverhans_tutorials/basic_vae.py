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

class ModelVAE(Model):
  def __init__(self, scope,latent_dim, **kwargs):
    del kwargs
    Model.__init__(self, scope, , n_hidden, locals())
    #self.n_fcc = n_fcc
    self.latent_dim = latent_dim
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
    xavier_initializer = tf.contrib.layers.xavier_initializer()
    
    conv1 = tf.layers.conv2d(inputs=x,
                             filters=3,
                             kernel_size=4,
                             strides=2,
                             padding='same',
                             kernel_initializer=xavier_initializer,
                             activation=tf.nn.relu)

    # Convolution outputs [batch, 8, 8, 64]
    conv2 = tf.layers.conv2d(inputs=conv1,
                               filters=64,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=xavier_initializer,
                               activation=tf.nn.relu)

    # Convolution outputs [batch, 4, 4, 64]
    conv3 = tf.layers.conv2d(inputs=conv2,
                               filters=64,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=xavier_initializer,
                               activation=tf.nn.relu)

    flat = tf.contrib.layers.flatten(conv3)

    z_mean = tf.layers.dense(flat, units=latent_dim, name='z_mean')
    z_log_var = tf.layers.dense(flat, units=latent_dim, name='z_log_var')

    samples = tf.random_normal([batch_size,latent_dim], 0, 1, dtype=tf.float32)
    z_flat = z_mean + (tf.exp(z_log_var) * samples)

    z_develop = tf.layers.dense(z_flat, units=4*4*64)

    net = tf.nn.relu(tf.reshape(z_develop, [-1, 4, 4, 64]))

    # Transposed convolution outputs [batch, 8, 8, 64]
    net = tf.layers.conv2d_transpose(inputs=net, 
                     filters=64,
                     kernel_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=xavier_initializer,
                     activation=tf.nn.relu)

    # Transposed convolution outputs [batch, 16, 16, 64]
    net = tf.layers.conv2d_transpose(inputs=net, 
                     filters=64,
                     kernel_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=xavier_initializer,
                     activation=tf.nn.relu)

    # Transposed convolution outputs [batch, 32, 32, 3]
    net = tf.layers.conv2d_transpose(inputs=net, 
                     filters=3,
                     kernel_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=xavier_initializer)
                                        

    net = tf.nn.sigmoid(net)


      
    return {
            'Z_FLAT': z_flat,
            'RECON': net
            }
