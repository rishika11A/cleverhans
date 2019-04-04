"""The CarliniWagnerL2 attack
"""
# pylint: disable=missing-docstring
import logging

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack_ae import Attack
from cleverhans.compat import reduce_sum, reduce_max
from cleverhans.model import CallableModelWrapper, Model, wrapper_warning_logits
from cleverhans import utils
import tensorflow.contrib.slim as slim  
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from tensorflow.python.framework import meta_graph
from keras import backend as K

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.carlini_wagner_l2")
_logger.setLevel(logging.INFO)


class CarliniWagnerAE_Keras(Attack):
 
  def __init__(self, model, cl_model, sess, dtypestr='float32', **kwargs):
    """
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
    if not isinstance(cl_model, Model):
      wrapper_warning_logits()
      #cl_model = CallableModelWrapper(cl_model, 'logits')
    if not isinstance(model, Model):
      wrapper_warning_logits()
      #model = CallableModelWrapper(model, 'logits')

    super(CarliniWagnerAE_Keras, self).__init__(model, sess, dtypestr, **kwargs)
    self.cl_model = cl_model
    self.feedable_kwargs = ('y', 'y_target')

    self.structural_kwargs = [
        'batch_size', 'confidence', 'targeted', 'learning_rate',
        'binary_search_steps', 'max_iterations', 'abort_early',
        'initial_const', 'clip_min', 'clip_max'
    ]

  def generate(self, x,x_t, **kwargs):
    
    assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
    self.parse_params(**kwargs)

    #labels, nb_classes = self.get_or_guess_labels(x, kwargs)
    nb_classes = 10
    attack = CWL2(self.sess, self.model, self.cl_model, self.batch_size, self.confidence,
                  'x_target' in kwargs, self.learning_rate,
                  self.binary_search_steps, self.max_iterations,
                  self.abort_early, self.initial_const, self.clip_min,
                  self.clip_max, nb_classes,
                  x.get_shape().as_list()[1:])

    def cw_wrap(x_val, x_targ_val):
      return np.array(attack.attack(x_val, x_targ_val), dtype=self.np_dtype)

    wrap = tf.py_func(cw_wrap, [x, x_t], self.tf_dtype)
    wrap.set_shape(x.get_shape())

    return wrap

  def parse_params(self,
                   y=None,
                   y_target = None,
                   x_target=None,
                   batch_size=1,
                   confidence=0,
                   learning_rate=5e-2,
                   binary_search_steps=10,
                   max_iterations=1000,
                   abort_early=False,
                   initial_const=0.5,
                   clip_min=0,
                   clip_max=1):
    

    # ignore the y and y_target argument
    self.batch_size = batch_size
    self.confidence = confidence
    self.learning_rate = learning_rate
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.abort_early = abort_early
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max


def ZERO():
  return np.asarray(0., dtype=np_dtype)

def convert_to_pb(weight_file,input_fld='',output_fld='', model_type = None):

    import os
    import os.path as osp
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from keras.models import load_model
    


    # weight_file is a .h5 keras model file
    output_node_names_of_input_network = ["pred0"] 
    output_node_names_of_final_network = 'output_node'

    # change filename to a .pb tensorflow file
    output_graph_name = weight_file[:-2]+'pb'
    weight_file_path = osp.join(input_fld, weight_file)
    '''
    if(model_type=='AE'):
      input_img = Input(shape=(32, 32, 3))
      x = Conv2D(64, (3, 3), padding='same')(input_img)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = MaxPooling2D((2, 2), padding='same')(x)
      x = Conv2D(32, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = MaxPooling2D((2, 2), padding='same')(x)
      x = Conv2D(16, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      encoded = MaxPooling2D((2, 2), padding='same')(x)

      x = Conv2D(16, (3, 3), padding='same')(encoded)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = UpSampling2D((2, 2))(x)
      x = Conv2D(32, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = UpSampling2D((2, 2))(x)
      x = Conv2D(64, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = UpSampling2D((2, 2))(x)
      x = Conv2D(3, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      decoded = Activation('sigmoid')(x)

      net_model = Model(input_img, decoded)
      net_model.load_weights(weight_file_path)

    if(model_type =='Classifier'):
      num_classes = 10
      net_model = Sequential()
      net_model.add(Conv2D(32, (3, 3), padding='same',
                       input_shape=(32,32,3)))
      net_model.add(Activation('relu'))
      net_model.add(Conv2D(32, (3, 3)))
      net_model.add(Activation('relu'))
      net_model.add(MaxPooling2D(pool_size=(2, 2)))
      net_model.add(Dropout(0.25))

      net_model.add(Conv2D(64, (3, 3), padding='same'))
      net_model.add(Activation('relu'))
      net_model.add(Conv2D(64, (3, 3)))
      net_model.add(Activation('relu'))
      net_model.add(MaxPooling2D(pool_size=(2, 2)))
      net_model.add(Dropout(0.25))

      net_model.add(Flatten())
      net_model.add(Dense(512))
      net_model.add(Activation('relu'))
      net_model.add(Dropout(0.5))
      net_model.add(Dense(num_classes))
      net_model.add(Activation('softmax'))

      opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
      net_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
      net_model.load_weights(weight_file_path)
    '''
    net_model = model.load(weight_file_path)
    #print("model.outputs: ", net_model.outputs)
    #print("model.inputs: ", net_model.inputs)
    num_output = len(output_node_names_of_input_network)
    pred = [None]*num_output
    pred_node_names = [None]*num_output

    for i in range(num_output):
        pred_node_names[i] = output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

    return output_fld+'/'+output_graph_name

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )

    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'

    return graph, input_name, output_name
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


class CWL2(object):
  def __init__(self, sess, model,cl_model, batch_size, confidence, targeted,
               learning_rate, binary_search_steps, max_iterations,
               abort_early, initial_const, clip_min, clip_max, num_labels,
               shape):
    
    self.sess = sess
    self.TARGETED = targeted
    self.LEARNING_RATE = learning_rate
    self.MAX_ITERATIONS = max_iterations
    self.BINARY_SEARCH_STEPS = binary_search_steps
    self.ABORT_EARLY = abort_early
    self.CONFIDENCE = confidence
    self.initial_const = initial_const
    self.batch_size = batch_size
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.model = model
    self.cl_model = cl_model

    #convert model to tensorflow model

    


    self.repeat = binary_search_steps >= 10

    self.shape = shape = tuple([batch_size] + list(shape))
    #print("shape: ", shape)

    # the variable we're going to optimize over
    modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))

    # these are variables to be more efficient in sending data to tf
    self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
    self.targimg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='targimg')
    #self.tlab = tf.Variable(
     #   np.zeros((batch_size, num_labels)), dtype=tf_dtype, name='tlab')
    self.const = tf.Variable(
        np.zeros(batch_size), dtype=tf_dtype, name='const')

    # and here's what we use to assign them
    self.assign_timg = tf.placeholder(tf_dtype, shape, name='assign_timg')
    self.assign_targimg = tf.placeholder(tf_dtype, shape, name='assign_targimg')
    #self.assign_tlab = tf.placeholder(
     #   tf_dtype, (batch_size, num_labels), name='assign_tlab')
    self.assign_const = tf.placeholder(
        tf_dtype, [batch_size], name='assign_const')

    # the resulting instance, tanh'd to keep bounded from clip_min
    # to clip_max
    self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
    self.newimg = self.newimg * (clip_max - clip_min) + clip_min

    #targimg_lat = latent_layer_model.predict(self.targimg)
    '''
    tf_model_path_ae = convert_to_pb('cifar10_AE.h5','../cleverhans_tutorials/models','../cleverhans_tutorials/models', 'AE')
    tf_model_path_cl = convert_to_pb('cifar10_CNN.h5','../cleverhans_tutorials/models','../cleverhans_tutorials/models', 'Classifier')
    tf_model,tf_input,tf_output = load_graph(tf_model_path_ae)
    tf_cl_model,tf_cl_input,tf_cl_output = load_graph(tf_model_path_cl)
    
    #self.x_hat = model.predict(self.newimg, steps = 1)
    with tf.Graph().as_default() as graph1:

      x_hat_output  = tf_model.get_tensor_by_name(tf_output) 
      x_hat_input = tf_model.get_tensor_by_name(tf_input)
      #self.x_hat_lat = latent_layer_model.predict(self.newimg)
      #self.x_hat = graph1.run(self.x_hat, feed_dict = {x_1 : self.newimg})
      #self.y_hat_logit = cl_model.predict(self.x_hat, steps = 1)
    with tf.Graph().as_default() as graph2:  
      y_hat_logit = tf_cl_model.get_tensor_by_name(tf_cl_output)
      y_hat_logit_input = tf_cl_model.get_tensor_by_name(tf_cl_input) 
      #self.y_hat_logit = self.sess.run(self.y_hat_logit, feed_dict = {x_2 : self.x_hat})
      #self.y_hat_logit = cl_model.predict(self.x_hat, steps = 1)

      y_hat_output = tf.argmax(y_hat_logit, axis = 1)

    x_1 = tf.placeholder(tf.float32, (None, 32,32, 3))
    graph = tf.get_default_graph()
    meta_graph1 = tf.train.export_meta_graph(graph=graph1)
    meta_graph.import_scoped_meta_graph(meta_graph1, input_map={'x_hat_input': x_1}, import_scope='graph1',
    out1 = graph.get_tensor_by_name('graph1/tf_output:0'))

    meta_graph2 = tf.train.export_meta_graph(graph=graph2)
    meta_graph.import_scoped_meta_graph(meta_graph2, input_map={'y_hat_logit_input': out1}, import_scope='graph2')
    #self.y_targ_logit = cl_model.predict(self.targimg, steps = 1)
    self.y_targ_logit = tf_cl_model.get_tensor_by_name(tf_cls_output)
    self.y_targ_logit = sess.run(self.y_targ_logit, feed_dict = {tf_cl_model.get_tensor_by_name(tf_cl_input): self.targimg})
    self.y_targ = tf.argmax(self.y_targ_logit, axis = 1)
    '''
    # distance to the input data

    #print("model.outputs: ", model.outputs)
    #print("model.inputs: ", model.inputs)
    frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "../cleverhans_tutorials/models", "tf_model_AE.pb", as_text=False)

    from tensorflow.python.platform import gfile

    f = gfile.FastGFile("../cleverhans_tutorials/models/tf_model_AE.pb", 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess.graph.as_default()
    tf.import_graph_def(graph_def)
    reconstruction_tensor = sess.graph.get_tensor_by_name('import/activation_7/Sigmoid:0')
    #self.x_hat = reconstruction_tensor(self.newimg)

    #self.y_hat_logit = cl_model.predict(self.x_hat, steps=1)
    #self.y_hat = tf.argmax(self.y_hat_logit, axis = 1)
    #self.x_hat = sess.run(reconstruction_tensor, {'import/input_1:0': self.newimg})
    self.other = (tf.tanh(self.timg) + 1) / 2
    self.other =  self.other * (clip_max - clip_min) + clip_min
    self.l2dist = reduce_sum(
        tf.square(self.newimg - self.other), list(range(1, len(shape))))

    print("shape of l2_dist: ", np.shape(self.l2dist))

    
    epsilon = 10e-8
    
    loss1 = reduce_sum(tf.square(self.x_hat-self.targimg))
    
    # sum up the losses
    self.loss2 = reduce_sum(self.l2dist)
    self.loss1 = reduce_sum(self.const * loss1)
    self.loss = self.loss1 + self.loss2

    # Setup the adam optimizer and keep track of variables we're creating
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
    self.train = optimizer.minimize(self.loss, var_list=[modifier])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]

    # these are the variables to initialize when we run
    self.setup = []
    self.setup.append(self.timg.assign(self.assign_timg))
    self.setup.append(self.targimg.assign(self.assign_targimg))
    #self.setup.append(self.tlab.assign(self.assign_tlab))
    self.setup.append(self.const.assign(self.assign_const))

    self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

  def attack(self, imgs, targ_imgs):
    
    print("batch_size in attack: ", self.batch_size)
    r = []
    for i in range(0, len(imgs), self.batch_size):
      _logger.debug(
          ("Running CWL2 attack on instance %s of %s", i, len(imgs)))
      r.extend(
          self.attack_batch(imgs[i:i + self.batch_size],
                            targ_imgs[i:i + self.batch_size]))
    return np.array(r)

  def attack_batch(self, imgs, targ_imgs):
    """
    Run the attack on a batch of instance and labels.
    """

    def compare(y1,y2):
      if(y1==y2):
        return True
      else:
        return False

    def compare_dist(recon, orig, targ):
      
      a = np.sum((recon - orig)**2)
      b = np.sum((recon-targ)**2)
      #if  (tf.math.greater(a,b)) :
      if(a+80>b):
        return True
      else:
        return False
    
    batch_size = self.batch_size

    #print("batch_size: ", batch_size)

    oimgs = np.clip(imgs, self.clip_min, self.clip_max)
    #oimgs_lat = self.model.get_layer(oimgs, 'LATENT')

    # re-scale instances to be within range [0, 1]
    imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
    imgs = np.clip(imgs, 0, 1)
    # now convert to [-1, 1]
    imgs = (imgs * 2) - 1
    # convert to tanh-space
    imgs = np.arctanh(imgs * .999999)

    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(batch_size)
    CONST = np.ones(batch_size) * self.initial_const
    upper_bound = np.ones(batch_size) * 1e8

    # placeholders for the best l2, score, and instance attack found so far
    o_bestl2 = [1e10] * batch_size
    #o_bestrec = np.copy(oimgs)
    o_bestrec = np.copy(oimgs)
    o_bestattack = np.copy(oimgs)

    for outer_step in range(self.BINARY_SEARCH_STEPS):
      # completely reset adam's internal state.
      self.sess.run(self.init)
      batch = imgs[:batch_size]
      batchtarg = targ_imgs[:batch_size]

      bestl2 = [1e10] * batch_size
      #bestrec = np.copy(oimgs)
      bestrec = np.copy(oimgs)

      _logger.debug("  Binary search step %s of %s",
                    outer_step, self.BINARY_SEARCH_STEPS)

      # The last iteration (if we run many steps) repeat the search once.
      if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
        CONST = upper_bound

      # set the variables so that we don't have to send them over again
      self.sess.run(
          self.setup, {
              self.assign_timg: batch,
              self.assign_targimg: batchtarg,
              self.assign_const: CONST
          })

      prev = 1e8
      for iteration in range(self.MAX_ITERATIONS):
        # perform the attack
        _, l, l2s, nrec, nimg, yhat, ytarg = self.sess.run([
            self.train, self.loss, self.l2dist, self.x_hat,
            self.newimg, self.y_hat, self.y_targ
        ])
        #print("shape of yhat: ", np.shape(yhat))
        if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
          _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                         "l2={:.3g} ").format(
                             iteration, self.MAX_ITERATIONS, l,
                             np.mean(l2s)))

        # check if we should abort search if we're getting nowhere.
        if self.ABORT_EARLY and \
           iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
          if l > prev * .9999:
            msg = "    Failed to make progress; stop early"
            _logger.debug(msg)
            break
          prev = l

        # adjust the best result found so far
        for e, (l2, nr, ii, yh, yt) in enumerate(zip(l2s, nrec, nimg, yhat, ytarg)):
          #lab = np.argmax(batchlab[e])
          if l2 < bestl2[e] and compare(yh,yt):
          #if l2<bestl2[e] and compare_dist(nr, imgs[e], targ_imgs[e]):
            bestl2[e] = l2
            bestrec[e] = nr
          if l2 < o_bestl2[e] and compare(yh,yt):
          #if l2 < o_bestl2[e] and compare_dist(nr, imgs[e], targ_imgs[e]):
            o_bestl2[e] = l2
            o_bestrec[e] = nr
            o_bestattack[e] = ii

      # adjust the constant as needed
      for e in range(batch_size):
        if compare(yhat[e], ytarg[e]):
        #if compare_dist(nrec[e], imgs[e], targ_imgs[e]):
          upper_bound[e] = min(upper_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
        else:
          # failure, either multiply by 10 if no solution found yet
          #          or do binary search with the known upper bound
          lower_bound[e] = max(lower_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
          else:
            CONST[e] *= 10
      _logger.debug("  Successfully generated adversarial examples " +
                    "on {} of {} instances.".format(
                        sum(upper_bound < 1e9), batch_size))
      o_bestl2 = np.array(o_bestl2)
      mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
      _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

    # return the best solution found
    o_bestl2 = np.array(o_bestl2)
    #print("o_bestl2: ", o_bestl2)
    #print("shape of o_bestattack: ", np.shape(o_bestattack))
    return o_bestattack
