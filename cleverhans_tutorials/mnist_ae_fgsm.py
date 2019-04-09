from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf

#from cleverhans.compat import flags
from tensorflow.python.platform import flags
from cleverhans.loss import CrossEntropy
from cleverhans.loss import SquaredError
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.utils_tf import model_eval_ae
#from cleverhans.train import train
from cleverhans.train_ae import train_ae
from cleverhans.train_cls import train_cls
#from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks.fast_gradient_method import FastGradientMethodAe
from cleverhans.utils import AccuracyReport, set_log_level, grid_visual
from cleverhans.model_zoo.basic_ae import ModelBasicAE
from cleverhans.model_zoo.basic_cls import ModelCls
import random
from skimage.filters.rank import mean
from skimage.morphology import disk
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter

import pylab as plt
import matplotlib
matplotlib.use('Agg')

from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
import os

FLAGS = flags.FLAGS

NB_EPOCHS = 8
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
adversarial_training = False
mean_filtering = True
binarization = True
#NB_FILTERS = 64


def mnist_ae(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE,
                   clean_train=CLEAN_TRAIN,
                   testing=False,
                   backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                   num_threads=None,
                   label_smoothing=0.1):
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get MNIST data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]
  nb_layers = 500
  nb_latent_size = 100
  source_samples = 10

  # Define input TF placeholder
  x = tf.placeholder( tf.float32, shape=(None, img_rows, img_cols, nchannels))
  x_t = tf.placeholder( tf.float32, shape=(None, img_rows, img_cols, nchannels))
  #r = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y = tf.placeholder( tf.float32, shape=(None, nb_classes))
  y_t = tf.placeholder( tf.float32, shape=(None, nb_classes))
  #set target images
  #print("np.shape(y_train): ", np.shape(y_train))
  #print(y_train[5])
  train_params_cls = {
      'nb_epochs': 15,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  rng = np.random.RandomState()
  eval_params_cls = {'batch_size': batch_size}

  class_model = ModelCls('model_classifier')

  def do_eval_cls(preds, x_set, y_set, x_tar_set,report_key, is_adv = None):
    acc = model_eval(sess, x, y, preds, x_t, x_set, y_set, x_tar_set, args=eval_params_cls)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))


  def eval_cls():
    do_eval_cls(y_logits,x_test,y_test,x_test,'clean_train_clean_eval', False)


  y_logits = class_model.get_layer(x,'LOGITS')
  loss_cls = CrossEntropy(class_model)
  
  train_cls(sess,loss_cls, x_train, y_train, evaluate = eval_cls,
                  args=train_params_cls, rng=rng, var_list=class_model.get_params())

  #x_train_target = tf.random_shuffle(x_train)
  #x_test_target = tf.random_shuffle(x_test)
  #x_train_target = x_train.copy()
  #x_test_target = x_test.copy()
  
  index_shuf = list(range(len(x_train)))
      # Randomly repeat a few training examples each epoch to avoid
      # having a too-small batch
  while len(index_shuf) % batch_size != 0:
    index_shuf.append(rng.randint(len(x_train)))
    nb_batches = len(index_shuf) // batch_size
    rng.shuffle(index_shuf)
    # Shuffling here versus inside the loop doesn't seem to affect
    # timing very much, but shuffling here makes the code slightly
    # easier to read
    x_train_target = x_train[index_shuf]
    y_train_target = y_train[index_shuf]
  

  for ind in range (0, len(x_train)):
    r_ind = -1
    while(np.argmax(y_train_target[ind])==np.argmax(y_train[ind])):
      r_ind = rng.randint(0,len(x_train))
      y_train_target[ind] = y_train[r_ind]
    if r_ind>-1:  
      x_train_target[ind] = x_train[r_ind]

  idxs = [np.where(np.argmax(y_test, axis=1) == i)[0][0] for i in range(nb_classes)]
  adv_inputs = np.array(
          [[instance] * (nb_classes-1) for instance in x_test[idxs]],
          dtype=np.float32)

  grid_shape = (nb_classes, nb_classes, img_rows, img_cols,
                    nchannels)
  grid_viz_data = np.zeros(grid_shape, dtype='f')
  grid_viz_data_1 = np.zeros(grid_shape, dtype='f')

  adv_input_y = []
  for curr_num in range(nb_classes):
    targ = []
    for id in range(nb_classes-1):
        targ.append(y_test[idxs[curr_num]])
    adv_input_y.append(targ)
  
  adv_input_y = np.array(adv_input_y)

  adv_target_y = []
  for curr_num in range(nb_classes):
    targ = []
    for id in range(nb_classes):
      if(id!=curr_num):
        targ.append(y_test[idxs[id]])
    adv_target_y.append(targ)
  
  adv_target_y = np.array(adv_target_y)

  #print("adv_input_y: \n", adv_input_y)
  #print("adv_target_y: \n", adv_target_y)

  adv_input_targets = []
  for curr_num in range(nb_classes):
    targ = []
    for id in range(nb_classes):
      if(id!=curr_num):
        targ.append(x_test[idxs[id]])
    adv_input_targets.append(targ)
  adv_input_targets = np.array(adv_input_targets)

  adv_inputs = adv_inputs.reshape(
    (source_samples * (nb_classes-1), img_rows, img_cols, nchannels))
  adv_input_targets = adv_input_targets.reshape(
    (source_samples * (nb_classes-1), img_rows, img_cols, nchannels))

  adv_input_y = adv_input_y.reshape(source_samples*(nb_classes-1), 10)
  adv_target_y = adv_target_y.reshape(source_samples*(nb_classes-1), 10)

  # Use Image Parameters
 

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  eval_params = {'batch_size': batch_size}
  fgsm_params = {
      'eps': 0.3,
      'clip_min': 0.,
      'clip_max': 1.
  }
  rng = np.random.RandomState([2017, 8, 30])
  '''
  def mnist_dist_diff(r, x, x_t):
    d1 = tf.reduce_sum(tf.squared_difference(r, x))
    d2 = tf.reduce_sum(tf.squared_difference(r, x_t))
    diff = d1-d2
    #sess_temp = tf.Session()
    #with sess_temp.as_default():
      #return diff.eval()
    return diff
  '''    
  def plot_results( adv_inputs, adv, recon_orig, recon_adv):
    nb_classes = 10
    img_rows = img_cols = 28
    nchannels = 1

    grid_shape = (nb_classes, nb_classes, img_rows, img_cols,
                    nchannels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')
    grid_viz_data_1 = np.zeros(grid_shape, dtype='f')
    curr_class = 0
    for j in range(nb_classes):
        for i in range(nb_classes):
          #grid_viz_data[i, j] = adv[j * (nb_classes-1) + i]
          if(i==j):
            grid_viz_data[i,j] = recon_orig[curr_class*9]
            grid_viz_data_1[i,j] = adv_inputs[curr_class*9]
            curr_class = curr_class+1
          else:
            if(j>i):
              grid_viz_data[i,j] = recon_adv[i*(nb_classes-1) + j-1]
              grid_viz_data_1[i,j] = adv[i*(nb_classes-1)+j-1]
            else:
              grid_viz_data[i,j] = recon_adv[i*(nb_classes-1) + j]
              grid_viz_data_1[i,j] = adv[i*(nb_classes-1)+j]

    _ = grid_visual(grid_viz_data)
    _ = grid_visual(grid_viz_data_1)
    
  def do_eval(recons, x_orig, x_target, y_orig, y_target, report_key, is_adv=False, x_adv = None, recon_adv = False, lat_orig = None, lat_orig_recon = None):
    #acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    #calculate l2 dist between (adv img, orig img), (adv img, target img), 
    #dist_diff = mnist_dist_diff(recons, x_orig, x_target)
    #problem : doesn't work for x, x_t
    noise, d_orig, d_targ, avg_dd, d_latent = model_eval_ae(sess, x, x_t, recons, x_orig, x_target, x_adv, recon_adv, lat_orig, lat_orig_recon, args = eval_params)
    
    setattr(report, report_key, avg_dd)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test d1 on ', report_text,  ' examples: ', d_orig)
      print('Test d2 on ', report_text,' examples: ', d_targ)
      print('Test distance difference on %s examples: %0.4f' % (report_text, avg_dd))
      print('Noise added: ', noise)
      print("dist_latent_orig_recon on ", report_text, "examples : ", d_latent)
      print()

    
  if clean_train:
    #model = ModelBasicCNN('model1', nb_classes, nb_filters)
    model = ModelBasicAE('model1', nb_layers,nb_latent_size )
    
    #preds = model.get_logits(x)
    recons = model.get_layer(x,'RECON')
    #tf.reshape(recons, (tf.shape(recons)[0],28,28))
    #loss = CrossEntropy(model, smoothing=label_smoothing)
    #loss = squared loss between x and recons
    #loss = tf.squared_difference(tf.reshape(x,(128,28*28)), recons)
    loss = SquaredError(model)
    

    latent1_orig = model.get_layer(x, 'LATENT')
    latent1_orig_recon = model.get_layer(recons, 'LATENT')
    print("np.shape(latent_orig): ",np.shape(latent1_orig))
    #y_logits = class_model.get_logits(latent1_orig)

    def evaluate():
      do_eval(recons, x_test, x_test, y_test, y_test, 'clean_train_clean_eval', False, None, None, latent1_orig, latent1_orig_recon)
    

    train_ae(sess, loss, x_train,x_train, evaluate=evaluate,
          args=train_params, rng=rng, var_list=model.get_params())

    
    #commented out
    #if testing:
     # do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    fgsm = FastGradientMethodAe(model, sess=sess)
    adv_x = fgsm.generate(x,x_t, **fgsm_params)
    #adv_x = fgsm.generate(adv_inputs,adv_input_targets, **fgsm_params)
    recons_adv = model.get_layer(adv_x, 'RECON')
    pred_adv = class_model.get_layer(adv_x, 'LOGITS')
    latent1_adv = model.get_layer(adv_x, 'LATENT')
    latent1_adv_recon = model.get_layer(recons_adv, 'LATENT')
    #dist_latent_adv_model1 = tf.reduce_sum(tf.squared_difference(latent1_adv, latent1_adv_recon))
    #dist_latent_orig_model1 = tf.reduce_sum(tf.squared_difference(latent1_orig, latent1_orig_recon))
    adv_evald = sess.run(adv_x, feed_dict = {x: adv_inputs, x_t: adv_input_targets})
    recons_orig = model.get_layer(adv_inputs, 'RECON')
    recons_orig_evald = sess.run(recons_orig, feed_dict = {x: adv_inputs})
    recons_adv_evald = sess.run(model.get_layer(adv_evald,'RECON'))
    #tf.reshape(recons_adv, (tf.shape(recons_adv)[0],28,28))
    # Evaluate the accuracy of the MNIST model on adversarial examples
    do_eval(recons_adv, adv_inputs, adv_input_targets, adv_input_y, adv_target_y, 'clean_train_adv_eval', True, adv_x, recons_adv, latent1_adv, latent1_adv_recon)
    do_eval_cls(pred_adv,adv_inputs,adv_target_y, adv_input_targets, 'clean_train_adv_eval', True)
    do_eval_cls(pred_adv,adv_inputs,adv_input_y, adv_input_targets, 'clean_train_adv_eval', True)
    #plot_results(adv_inputs, adv, recons_orig, recons_adv, False)
    plot_results(adv_inputs, adv_evald, recons_orig_evald, recons_adv_evald)

  
    #plot_results(sess, x_test[0:5], x_test_target[0:5], recons[0:5], adv_x[0:5], recons_adv[0:5], adv_trained = False)

    # Calculate training error
    if testing:
      do_eval(recons, x_train, x_train_target, y_train, y_train_target, 'train_clean_train_adv_eval', False)
      

    print('Repeating the process, using adversarial training')
    print()
    # Create a new model and train it to be robust to FastGradientMethod
  
  if(adversarial_training == True):
    model2 = ModelBasicAE('model2', nb_layers, nb_latent_size)
    fgsm2 = FastGradientMethodAe(model2, sess=sess)

    def attack(x, x_t):

      return fgsm2.generate(x, x_t, **fgsm_params)

    #loss2 = CrossEntropy(model2, smoothing=label_smoothing, attack=attack)
    #loss2 = squared loss b/w x_orig and adv_recons
    loss2 = SquaredError(model2, attack = attack)
    adv_x2 = attack(x, x_t)
    recons2 = model2.get_layer(x, 'RECON')
    pred_adv2 = class_model.get_layer(adv_x2, 'LOGITS')
    #adv_noise = adv_x2 - x
    
    

    if not backprop_through_attack:
    # For the fgsm attack used in this tutorial, the attack has zero
    # gradient so enabling this flag does not change the gradient.
    # For some other attacks, enabling this flag increases the cost of
    # training, but gives the defender the ability to anticipate how
    # the atacker will change their strategy in response to updates to
    # the defender's parameters.
      adv_x2 = tf.stop_gradient(adv_x2)
    recons2_adv = model2.get_layer(adv_x2, 'RECON')

    latent2_orig = model2.get_layer(x, 'LATENT')
    latent2_orig_recon = model2.get_layer(recons2, 'LATENT')
    latent2_adv = model2.get_layer(adv_x2, 'LATENT')
    latent2_adv_recon = model2.get_layer(recons2_adv, 'LATENT')
    #dist_latent_adv_model2 = tf.reduce_sum(tf.squared_difference(latent2_adv, latent2_adv_recon))
    #dist_latent_orig_model2 = tf.reduce_sum(tf.squared_difference(latent2_orig, latent2_orig_recon))
    recons_orig = model2.get_layer(adv_inputs, 'RECON')
    

    def evaluate2():
      # Accuracy of adversarially trained model on legitimate test inputs
      do_eval(recons2, x_test, x_test, y_test, y_test, 'adv_train_clean_eval', False, None, None, latent2_orig, latent2_orig_recon)
      # Accuracy of the adversarially trained model on adversarial examples
      do_eval(recons2_adv, adv_inputs, adv_input_targets, adv_input_y, adv_target_y,  'adv_train_adv_eval', True, adv_x2, recons2_adv, latent2_adv, latent2_adv_recon)
      do_eval_cls(pred_adv2, adv_inputs, adv_target_y, adv_input_targets,'adv_train_adv_eval', True)
      do_eval_cls(pred_adv2,adv_inputs,adv_input_y, adv_input_targets,'adv_train_adv_eval', True)
      #plot_results(x, x_t,recons2, adv_x2, recons2_adv, True, adv_inputs, adv_input_targets)
      
    # Perform and evaluate adversarial training
    train_ae(sess, loss2, x_train, x_train_target, evaluate=evaluate2,
          args=train_params, rng=rng, var_list=model2.get_params())
    
    
    adv_evald = sess.run(adv_x2, feed_dict = {x: adv_inputs, x_t: adv_input_targets})
    recons_adv_evald = sess.run(model2.get_layer(adv_evald, 'RECON'))
    recons_orig_evald = sess.run(recons_orig, feed_dict = {x: adv_inputs})

    plot_results(adv_inputs, adv_evald, recons_orig_evald, recons_adv_evald)

    # Calculate training errors
    if testing:
      do_eval(recons2, x_train, x_train,y_train, y_train,'train_adv_train_clean_eval', False)
      do_eval(recons2_adv, x_train, x_train_target, y_train, y_train_target,'train_adv_train_adv_eval', True, adv_x2, recons2_adv, latent2_adv, latent2_adv_recon)
      do_eval_cls(pred_adv2, adv_inputs, adv_target_y, adv_input_targets, 'train_adv_train_adv_eval', True)
      do_eval_cls(pred_adv2,adv_inputs,adv_input_y, adv_input_targets, 'train_adv_train_adv_eval', True)
      #plot_results(sess, x_train[0:5], x_train_target[0:5], recons2[0:5], adv_x2[0:5], recons2_adv[0:5], adv_trained = True)
  
  if (binarization == True):
    print("binarization")
    print("-------------")
    adv_evald[adv_evald>0.5] = 1.0
    adv_evald[adv_evald<=0.5] = 0.0

    recon_adv = model.get_layer(adv_evald, 'RECON')
    lat_orig = model.get_layer(x, 'LATENT')
    lat_orig_recon = model.get_layer(recons, 'LATENT')
    pred_adv_recon = class_model.get_layer(recon_adv, 'LOGITS')
    eval_params = {'batch_size': 90}

    recon_adv = sess.run(recon_adv)
    pred_adv_recon = sess.run(pred_adv_recon)
    #noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recons, adv_inputs, adv_input_targets, adv_evald, recon_adv,lat_orig, lat_orig_recon, args=eval_params)
    noise = np.sum(np.square(adv_evald-adv_inputs))/len(adv_inputs)
    noise = pow(noise,0.5)
    d1 = np.sum(np.square(recon_adv-adv_inputs))/len(adv_inputs)
    d2 = np.sum(np.square(recon_adv-adv_input_targets))/len(adv_inputs)
    acc1 = (sum(np.argmax(pred_adv_recon, axis=-1)==np.argmax(adv_target_y, axis=-1)))/len(adv_inputs)
    acc2 = (sum(np.argmax(pred_adv_recon, axis=-1)==np.argmax(adv_input_y, axis=-1)))/len(adv_inputs)
    print("d1: ", d1)
    print("d2: ", d2)
    print("noise: ", noise)
    print("classifier acc for target class: ", acc1)
    print("classifier acc for true class: ", acc2)
    #do_eval_cls(pred_adv_recon,adv_inputs,adv_input_y, adv_input_targets, 'clean_train_adv_eval', True)
    #do_eval_cls(pred_adv_recon,adv_inputs,adv_target_y, adv_input_targets, 'clean_train_adv_eval', True)
    #print("classifier acc for target class: ", acc1)
    #print("classifier acc for true class: ", acc2)
    plot_results(adv_inputs, adv_evald, recons_orig_evald, recon_adv)


  if (mean_filtering == True):
    print("mean filtering")
    print("--------------------")
    adv_evald = uniform_filter(adv_evald, 2)
    recon_adv = model.get_layer(adv_evald, 'RECON')
    lat_orig = model.get_layer(x, 'LATENT')
    lat_orig_recon = model.get_layer(recons, 'LATENT')
    pred_adv_recon = class_model.get_layer(recon_adv, 'LOGITS')
    eval_params = {'batch_size': 90}
    
    recon_adv = sess.run(recon_adv)
    pred_adv_recon = sess.run(pred_adv_recon)
    
    noise = np.sum(np.square(adv_evald-adv_inputs))/len(adv_inputs)
    noise = pow(noise,0.5)
    d1 = np.sum(np.square(recon_adv-adv_inputs))/len(adv_inputs)
    d2 = np.sum(np.square(recon_adv-adv_input_targets))/len(adv_inputs)
    acc1 = (sum(np.argmax(pred_adv_recon, axis=-1)==np.argmax(adv_target_y, axis=-1)))/len(adv_inputs)
    acc2 = (sum(np.argmax(pred_adv_recon, axis=-1)==np.argmax(adv_input_y, axis=-1)))/len(adv_inputs)

    print("d1: ", d1)
    print("d2: ", d2)
    print("noise: ", noise)
    print("classifier acc for target class: ", acc1)
    print("classifier acc for true class: ", acc2)
    plot_results(adv_inputs, adv_evald, recons_orig_evald, recon_adv)
    
  return report
def main(argv=None):
  """
  Run the tutorial using command line flags.
  """
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_ae(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 clean_train=FLAGS.clean_train,
                 backprop_through_attack=FLAGS.backprop_through_attack)
               


if __name__ == '__main__':
  #flags.DEFINE_integer('nb_filters', NB_FILTERS,
   #                    'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))

  tf.app.run()
  #`run()