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
from cleverhans.utils_tf import model_eval_full
#from cleverhans.train import train
from cleverhans.train_ae import train_ae
from cleverhans.train_cls import train_cls
#from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks.fast_gradient_method import FastGradientMethodAe
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model_zoo.basic_ae import ModelBasicAE
from cleverhans.model_zoo.basic_cls import ModelCls
import random

import pylab as plt
import matplotlib
matplotlib.use('Agg')

from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
import os

FLAGS = flags.FLAGS

NB_EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
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

  # Define input TF placeholder
  x = tf.placeholder( tf.float32, shape=(None, img_rows, img_cols, nchannels))
  x_t = tf.placeholder( tf.float32, shape=(None, img_rows, img_cols, nchannels))
  #r = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y = tf.placeholder( tf.float32, shape=(None, nb_classes))
  y_t = tf.placeholder( tf.float32, shape=(None, nb_classes))
  z = tf.placeholder(tf.float32, shape = (None, 100))
  z_t = tf.placeholder(tf.float32, shape = (None, 100))
  #set target images
  #print("np.shape(y_train): ", np.shape(y_train))
  #print(y_train[5])
 

  #y_logits = class_model.get_layer(x,'LOGITS')
  
  
  #y_pred = class_model.get_layer(x,'PRED')
  
  
  #x_train_target = tf.random_shuffle(x_train)
  #x_test_target = tf.random_shuffle(x_test)
  #x_train_target = x_train.copy()
  #x_test_target = x_test.copy()
  rng = np.random.RandomState()
  
  index_shuf = list(range(len(x_train)))
      # Randomly repeat a few training examples each epoch to avoid
      # having a too-small batch
  '''
  while len(index_shuf) % batch_size != 0:
    index_shuf.append(rng.randint(len(x_train)))
    nb_batches = len(index_shuf) // batch_size
    rng.shuffle(index_shuf)
    # Shuffling here versus inside the loop doesn't seem to affect
    # timing very much, but shuffling here makes the code slightly
    # easier to read
    x_train_target= x_train[index_shuf]
    y_train_target = y_train[index_shuf]
  '''
  rng.shuffle(index_shuf)
  x_train_target= x_train[index_shuf]
  y_train_target = y_train[index_shuf]
  
  '''  
  for ind in range (0, len(x_train)):
    r_ind = -1
    while(np.argmax(y_train_target[ind])==np.argmax(y_train[ind])):
      r_ind = rng.randint(0,len(x_train))
      y_train_target[ind] = y_train[r_ind]
    if r_ind>-1:  
      x_train_target[ind] = x_train[r_ind]
  '''
  index_shuf = list(range(len(x_test)))
  '''
  while len(index_shuf) % batch_size != 0:
    index_shuf.append(rng.randint(len(x_test)))
    nb_batches = len(index_shuf) // batch_size
    rng.shuffle(index_shuf)
    # Shuffling here versus inside the loop doesn't seem to affect
    # timing very much, but shuffling here makes the code slightly
    # easier to read
    x_test_target= x_test[index_shuf]
    y_test_target = y_test[index_shuf]
  '''
  rng.shuffle(index_shuf)
  x_test_target= x_test[index_shuf]
  y_test_target = y_test[index_shuf]
  '''
  for ind in range (0, len(x_test)):
    r_ind = -1
    while(np.argmax(y_test_target[ind])==np.argmax(y_test[ind])):
      r_ind = rng.randint(0,len(x_test))
      y_test_target[ind] = y_test[r_ind]
    if r_ind>-1:
      x_test_target[ind] = x_test[r_ind]
  '''
  # Use Image Parameters
  print("shape of x_train: ",np.shape(x_train))
  print("shape of x_train_target: ", np.shape(x_train_target))
  print("shape of x_test: ", np.shape(x_test))
  print("shape of x_test_target: ", np.shape(x_test_target))


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
  def plot_results( x_orig, x_targ, recon, adv_x, recon_adv, adv_trained, X_orig = None, X_targ = None):

    start = 0
    end = 10
    cur_batch_size = 10

    #global _model_eval_cache
    #args = _ArgsWrapper(args or {})

    '''
    print("np.shape(X_orig): ", np.shape(X_orig))
    print("type(X_orig): ",type(X_orig))
    '''

    with sess.as_default():

      l1 = np.shape(x_orig)
      l2 = np.shape(x_targ)
      X_cur = np.zeros((cur_batch_size,l1[1],l1[2], l1[3]), dtype='float64')
      X_targ_cur = np.zeros((cur_batch_size,l2[1], l2[2], l2[3]),dtype='float64')

      X_cur[:cur_batch_size] = X_orig[start:end]
      X_targ_cur[:cur_batch_size] = X_targ[start:end]

      feed_dict_1 = {x_orig: X_cur, x_targ: X_targ_cur}
      recon = np.squeeze(recon.eval(feed_dict=feed_dict_1))
      adv_x = np.squeeze(adv_x.eval(feed_dict=feed_dict_1))
      recon_adv = np.squeeze(recon_adv.eval(feed_dict=feed_dict_1))
      
      #x_orig = (np.squeeze(x_orig)).astype(float)
      #x_targ = (np.squeeze(x_targ)).astype(float)
     
      #adv_trained = tf.to_float(tf.squeeze(adv_trained.eval()))
      '''
      print("np.shape(x_orig): ", np.shape(x_orig))
      print("np.shape(recon): ", np.shape(recon))
      print("type(x_orig): ",type(x_orig))
      print("type(recon): ",type(recon))
      '''

      for i in range (0,8):
        
        fig = plt.figure(figsize=(9,6))
        
        img = X_cur[i]
        img = np.squeeze(X_cur[i]).astype(float)
        #tf.to_float(img)
        title = "Original Image"
        #img = img.reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(2, 3, 1)
        #Image.fromarray(np.asarray(img)).show()
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")
        
        img = recon[i]
        title = "Recon (Original Image)"
        #img = img.reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(2, 3, 2)
        plt.imshow(img, cmap = 'Greys_r')
        plt.title(title)
        plt.axis("off")

        img = X_targ_cur[i]
        img = np.squeeze(X_targ_cur[i]).astype(float)
        title = "Target Image"
        #img = img.reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(2, 3, 3)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")
      
        img = adv_x[i]-np.squeeze(X_targ_cur[i]).astype(float)
        title = "Noise added"
        #img = img.reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(2, 3, 4)
        plt.imshow(img, cmap = 'Greys_r')
        plt.title(title)
        plt.axis("off")


        img = adv_x[i]
        title = "Adv Image"
        #img = img.reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(2, 3, 5)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")

        img = recon_adv[i]
        title = "Recon (Adv Image)"
        #img = img.reshape(28, 28)
        img = np.clip(img, 0, 1)
        plt.subplot(2, 3, 6)
        plt.imshow(img, cmap='Greys_r')
        plt.title(title)
        plt.axis("off")

        output_dir = 'results/adv_mnist_ae/'
        if(adv_trained is False):
          fig.savefig(os.path.join(output_dir, ('results_' + str(i)+ '.png')))
        else:
          fig.savefig(os.path.join(output_dir, ('adv_tr_'  + str(i)+ '.png')))
        plt.close(fig)

  def do_eval(recons, x_orig, x_target, y_orig, y_target, report_key, is_adv=False, x_adv = None, recon_adv = False, lat_orig = None, lat_orig_recon = None):
    #acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    #calculate l2 dist between (adv img, orig img), (adv img, target img), 
    #dist_diff = mnist_dist_diff(recons, x_orig, x_target)
    #problem : doesn't work for x, x_t
    noise, d_orig, d_targ, avg_dd, d_latent = model_eval_ae(sess, x, x_t, y, y_t, recons, x_orig, x_target, y_orig, y_target, x_adv, recon_adv, lat_orig, lat_orig_recon, args = eval_params)
    
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

  
  train_params_cls = {
      'nb_epochs': 12,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
 
  eval_params_cls = {'batch_size': batch_size}

  class_model = ModelCls('model_classifier')

  def do_eval_cls(preds, z_set, y_set, z_tar_set,report_key, is_adv = None):
    acc = model_eval(sess, z, y, preds, z_t, z_set, y_set, z_tar_set, args=eval_params_cls)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  def do_eval_cls_full(preds, z_set, y_set, z_tar_set,report_key, is_adv = None):
    acc = model_eval_full(sess, z, y, preds, z_t, z_set, y_set, z_tar_set, args=eval_params_cls)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  
  
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
    loss_cls = CrossEntropy(class_model)

    latent1_orig = model.get_layer(x, 'LATENT')
    latent1_orig_recon = model.get_layer(recons, 'LATENT')
    print("np.shape(latent_orig): ",np.shape(latent1_orig))
    

    def evaluate():
      do_eval(recons, x_test, x_test, y_test, y_test, 'clean_train_clean_eval', False, None, None, latent1_orig, latent1_orig_recon)
    

    train_ae(sess, loss, x_train,x_train, evaluate=evaluate,
          args=train_params, rng=rng, var_list=model.get_params())

    y_logits = class_model.get_logits(z)
    feed_dict_a = {x : x_train}
    feed_dict_b = {x: x_test}
    feed_dict_c = {x: x_test_target}
    latent_orig_train = latent1_orig.eval(session =sess, feed_dict = feed_dict_a)
    latent_orig_test = latent1_orig.eval(session = sess, feed_dict = feed_dict_b)
    latent_target_test = latent1_orig.eval(session = sess, feed_dict = feed_dict_c)

    def eval_cls():
      do_eval_cls(y_logits,latent_orig_test,y_test,latent_orig_test,'clean_train_clean_eval', False)

    train_cls(sess,loss_cls, latent_orig_train, y_train, evaluate = eval_cls,
                  args=train_params_cls, rng=rng, var_list=class_model.get_params())

    #commented out
    #if testing:
     # do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    fgsm = FastGradientMethodAe(model, sess=sess)
    adv_x = fgsm.generate(x,x_t, **fgsm_params)
    recons_adv = model.get_layer(adv_x, 'RECON')
    latent1_adv = model.get_layer(adv_x, 'LATENT')
    latent1_adv_recon = model.get_layer(recons_adv, 'LATENT')
    feed_dict_adv = {x: x_test, x_t: x_test_target}
    #adv_x_evald = adv_x.eval(session = sess, feed_dict = feed_dict_adv)
    tf.global_variables_initializer().eval(session = sess)
    latent_adv = latent1_adv.eval(session = sess, feed_dict = feed_dict_adv)

    pred_adv = class_model.get_layer(latent_adv, 'LOGITS')
    
    
    dist_latent_adv_model1 = tf.reduce_sum(tf.squared_difference(latent1_adv, latent1_adv_recon))
    dist_latent_orig_model1 = tf.reduce_sum(tf.squared_difference(latent1_orig, latent1_orig_recon))

    #tf.reshape(recons_adv, (tf.shape(recons_adv)[0],28,28))
    # Evaluate the accuracy of the MNIST model on adversarial examples
    do_eval(recons_adv, x_test, x_test_target, y_test, y_test_target, 'clean_train_adv_eval', True, adv_x, recons_adv, latent1_adv, latent1_adv_recon)
    do_eval_cls_full(pred_adv,latent_orig_test,y_test_target, latent_target_test, 'clean_train_adv_eval', True)
    do_eval_cls_full(pred_adv,latent_orig_test,y_test, latent_target_test, 'clean_train_adv_eval', True)
    plot_results(x, x_t,recons, adv_x, recons_adv, False, x_test, x_test_target)
  
    #plot_results(sess, x_test[0:5], x_test_target[0:5], recons[0:5], adv_x[0:5], recons_adv[0:5], adv_trained = False)

    # Calculate training error
    if testing:
      do_eval(recons, x_train, x_train_target, y_train, y_train_target, 'train_clean_train_adv_eval', False)
      

    print('Repeating the process, using adversarial training')
    print()
    # Create a new model and train it to be robust to FastGradientMethod
  model2 = ModelBasicAE('model2', nb_layers, nb_latent_size)
  fgsm2 = FastGradientMethodAe(model2, sess=sess)

  def attack(x, x_t):

    return fgsm2.generate(x, x_t, **fgsm_params)

  #loss2 = CrossEntropy(model2, smoothing=label_smoothing, attack=attack)
  #loss2 = squared loss b/w x_orig and adv_recons
  loss2 = SquaredError(model2, attack = attack)
  adv_x2 = attack(x, x_t)
  recons2 = model2.get_layer(x, 'RECON')

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
  #adv_x2_evald = adv_x2.eval(session = sess, feed_dict = feed_dict_adv)
  #feed_dict_d = {x: adv_x2_evald}
  tf.global_variables_initializer().eval(session = sess)
  latent_adv = latent2_adv.eval(session = sess, feed_dict = feed_dict_adv)
  pred_adv2 = class_model.get_layer(latent_adv, 'LOGITS')

  dist_latent_adv_model2 = tf.reduce_sum(tf.squared_difference(latent2_adv, latent2_adv_recon))
  dist_latent_orig_model2 = tf.reduce_sum(tf.squared_difference(latent2_orig, latent2_orig_recon))

  def evaluate2():
    # Accuracy of adversarially trained model on legitimate test inputs
    do_eval(recons2, x_test, x_test, y_test, y_test, 'adv_train_clean_eval', False, None, None, latent2_orig, latent2_orig_recon)
    # Accuracy of the adversarially trained model on adversarial examples
    do_eval(recons2_adv, x_test, x_test_target, y_test, y_test_target,  'adv_train_adv_eval', True, adv_x2, recons2_adv, latent2_adv, latent2_adv_recon)
    do_eval_cls_full(pred_adv2,latent_orig_test,y_test_target, latent_target_test,'adv_train_adv_eval', True)
    do_eval_cls_full(pred_adv2,latent_orig_test,y_test, latent_target_test,'adv_train_adv_eval', True)
    plot_results(x, x_t,recons2, adv_x2, recons2_adv, True, x_test, x_test_target)

  # Perform and evaluate adversarial training
  train_ae(sess, loss2, x_train, x_train_target, evaluate=evaluate2,
        args=train_params, rng=rng, var_list=model2.get_params())
  
  
   
  # Calculate training errors
  if testing:
    do_eval(recons2, x_train, x_train,y_train, y_train,'train_adv_train_clean_eval', False)
    do_eval(recons2_adv, x_train, x_train_target, y_train, y_train_target,'train_adv_train_adv_eval', True, adv_x2, recons2_adv, latent2_adv, latent2_adv_recon)
    #do_eval_cls(pred_adv2,latent_orig_train,y_train_target,latent_target_train, 'train_adv_train_adv_eval', True)
    #do_eval_cls(pred_adv2,latent_orig_test,y_test,latent_target_test, 'train_adv_train_adv_eval', True)
    #plot_results(sess, x_train[0:5], x_train_target[0:5], recons2[0:5], adv_x2[0:5], recons2_adv[0:5], adv_trained = True)
    
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
  #sess.run()