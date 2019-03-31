"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import numpy as np
import tensorflow as tf

from cleverhans.attacks import CarliniWagnerAE
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.loss import SquaredError
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load, model_eval_ae
from cleverhans.train_ae import train_ae
from cleverhans.train_cls import train_cls
from cleverhans.train import train
from cleverhans.model_zoo.basic_ae import ModelBasicAE
from cleverhans.model_zoo.basic_cls import ModelCls
from skimage.filters.rank import mean
from skimage.morphology import disk
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
from cleverhans.dataset import CIFAR10
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.serial import save

FLAGS = flags.FLAGS

VIZ_ENABLED = True
BATCH_SIZE = 90
NB_EPOCHS = 6
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
CW_LEARNING_RATE = .4
ATTACK_ITERATIONS = 1000  
MODEL_PATH = os.path.join('models', 'mnist_cw')
MODEL_PATH_CLS = os.path.join('models', 'mnist_cw_cl')
TARGETED = True
adv_train = False
binarization_defense = False
mean_filtering = False
NB_FILTERS = 8 #64
clean_train = True

def cifar10_cw_recon(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=VIZ_ENABLED,
                      nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                      source_samples=SOURCE_SAMPLES,
                      learning_rate=LEARNING_RATE,
                      attack_iterations=ATTACK_ITERATIONS,
                      model_path=MODEL_PATH,
                      model_path_cls = MODEL_PATH,
                      targeted=TARGETED,
                      num_threads=None,
                      label_smoothing=0.1,
                      nb_filters=NB_FILTERS):
  
  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)
  rng = np.random.RandomState()


  # Create TF session
  sess = tf.Session()
  print("Created TensorFlow session.")

  set_log_level(logging.DEBUG)

  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get CIFAR10 data
  data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')

  nb_latent_size = 100
  # Get MNIST test data
  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]
  print("img_Rows, img_cols, nchannels: ", img_rows, img_cols, nchannels)

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  x_t = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  y_t = tf.placeholder( tf.float32, shape=(None, nb_classes))
  z = tf.placeholder(tf.float32, shape = (None, nb_latent_size))
  z_t = tf.placeholder(tf.float32, shape = (None, nb_latent_size))

  #nb_filters = 64
  nb_layers = 500
  
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
    do_eval_cls(y_logits, x_test, y_test, x_test,'clean_train_clean_eval', False)

  def evaluate():
        do_eval(y_logits, x_test, y_test, 'clean_train_clean_eval', False)

  filepath_ae = "clean_model_cifar10_ae.joblib"
  filepath_cl = "classifier_cifar10.joblib"

  if(clean_train==True):
  # Define TF model graph
    model = ModelBasicAE('model', nb_layers, nb_latent_size)
    #cl_model = ModelCls('cl_model')
    cl_model = ModelAllConvolutional('model1', nb_classes, nb_filters,
                                    input_shape=[32, 32, 3])
    #preds = model.get_logits(x)
    recons = model.get_layer(x, 'RECON')
    loss = SquaredError(model)
    print("Defined TensorFlow model graph.")
    y_logits = cl_model.get_logits(x)
    loss_cls = CrossEntropy(cl_model, smoothing=label_smoothing)
    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'filename': os.path.split(model_path)[-1]
    }
    
    train_params_cls = {
        'nb_epochs': 3,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    #if os.path.exists(model_path + ".meta"):
     # tf_model_load(sess, model_path)
    #else:
    eval_params_cls = {'batch_size': batch_size}
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    print("Training autoencoder")
    train_ae(sess, loss, x_train,x_train,args=train_params, rng=rng, var_list=model.get_params())
    #with sess.as_default():
     # save(filepath_ae, model)
    print("Training CNN")
    train(sess, loss_cls, None, None,
            dataset_train=dataset_train, dataset_size=dataset_size,
            evaluate=eval_cls, args=train_params_cls, rng=rng,
            var_list=cl_model.get_params())
    #with sess.as_default():
     # save(filepath_cl, cl_model)
  '''
  else:
    

    model = load(filepath_ae)
    cl_model = load(filepath_cl)
  '''

  #train_cls(sess, loss_cls, x_train, y_train, evaluate = eval_cls, args = train_params_cls, rng = rng, var_list = cl_model.get_params())
  #train_cls(sess, loss_cls, x_train, y_train, evaluate = eval_cls, args = train_params_cls, rng = rng, var_list = cl_model.get_params())
  
  ###########################################################################
  # Craft adversarial examples using Carlini and Wagner's approach
  ###########################################################################
  nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
  print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
        ' adversarial examples')
  print("This could take some time ...")

  # Instantiate a CW attack object
  cw = CarliniWagnerAE(model,cl_model, sess=sess)

  if viz_enabled:
    assert source_samples == nb_classes
    idxs = [np.where(np.argmax(y_test, axis=1) == i)[0][0]
            for i in range(nb_classes)]
  if targeted:
    if viz_enabled:
      # Initialize our array for grid visualization
      grid_shape = (nb_classes, nb_classes, img_rows, img_cols,
                    nchannels)
      grid_viz_data = np.zeros(grid_shape, dtype='f')
      grid_viz_data_1 = np.zeros(grid_shape, dtype='f')

      adv_inputs = np.array(
          [[instance] * (nb_classes-1) for instance in x_test[idxs]],
          dtype=np.float32)

      #adv_input_y = np.array([[instance]*(nb_classes-1) for instance in y_test[idxs]])
      
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

    one_hot = np.zeros((nb_classes, nb_classes))
    one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

  
  
  
  #saver.save(sess, model_path_cls)


    #adv_input_y = cl_model.get_layer(adv_inputs, 'LOGITS')
    #adv_target_y = cl_model.get_layer(adv_input_targets, 'LOGITS')

    

  adv_ys = np.array([one_hot] * source_samples,
                      dtype=np.float32).reshape((source_samples *
                                                 nb_classes, nb_classes))
  yname = "y_target"

  cw_params_batch_size = source_samples * (nb_classes-1)
  
  cw_params = {'binary_search_steps': 5,
               yname: adv_ys,
               'max_iterations': attack_iterations,
               'learning_rate': CW_LEARNING_RATE,
               'batch_size': cw_params_batch_size,
               'initial_const': 1}

  adv = cw.generate_np(adv_inputs, adv_input_targets,
                       **cw_params)

  #print("shaep of adv: ", np.shape(adv))
  recon_orig = model.get_layer(adv_inputs, 'RECON')
  recon_adv = model.get_layer(adv, 'RECON')
  lat_orig = model.get_layer(x, 'LATENT')
  lat_orig_recon = model.get_layer(recons, 'LATENT')
  pred_adv_recon = cl_model.get_logits(recon_adv)

  #eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
  eval_params = {'batch_size': 90}
  if targeted:
    noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recons, adv_inputs, adv_input_targets, adv, recon_adv,lat_orig, lat_orig_recon, args=eval_params)
    acc_1 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_target_y, adv_input_targets, args=eval_params_cls)
    acc_2 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_input_y, adv_input_targets, args=eval_params_cls)
    print("noise: ", noise)
    print("classifier acc_target: ", acc_1)
    print("classifier acc_true: ", acc_2)

  recon_adv = sess.run(recon_adv)
  recon_orig = sess.run(recon_orig)
  #print("recon_adv[0]\n", recon_adv[0,:,:,0])
  curr_class = 0
  if viz_enabled:
    for j in range(nb_classes):
      if targeted:
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


    #rint(grid_viz_data.shape)

  print('--------------------------------------')

  # Compute the number of adversarial examples that were successfully found

  # Compute the average distortion introduced by the algorithm
  percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                     axis=(1, 2, 3))**.5)
  print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

  # Close TF session
  #sess.close()

  # Finally, block & display a grid of all the adversarial examples
  
  if viz_enabled:
    _ = grid_visual(grid_viz_data)
    _ = grid_visual(grid_viz_data_1)
  
  #return report

  #adversarial training
  if(adv_train == True):

    
    print("starting adversarial training")
    #sess1 = tf.Session()
    adv_input_set = []
    adv_input_target_set = []

    for i in range(20):

      
      indices = np.arange(np.shape(x_train)[0])
      np.random.shuffle(indices)
      print("indices: ", indices[1:10])
      x_train = x_train[indices]
      y_train = y_train[indices]


      idxs = [np.where(np.argmax(y_train, axis=1) == i)[0][0] for i in range(nb_classes)]
      adv_inputs_2 = np.array(
            [[instance] * (nb_classes-1) for instance in x_train[idxs]],
            dtype=np.float32)
      adv_input_targets_2 = []
      for curr_num in range(nb_classes):
        targ = []
        for id in range(nb_classes):
          if(id!=curr_num):
            targ.append(x_train[idxs[id]])
        adv_input_targets_2.append(targ)
      adv_input_targets_2 = np.array(adv_input_targets_2)

      adv_inputs_2 = adv_inputs_2.reshape(
        (source_samples * (nb_classes-1), img_rows, img_cols, nchannels))
      adv_input_targets_2 = adv_input_targets_2.reshape(
        (source_samples * (nb_classes-1), img_rows, img_cols, nchannels))

      adv_input_set.append(adv_inputs_2)
      adv_input_target_set.append(adv_input_targets_2)

    adv_input_set = np.array(adv_input_set),
    adv_input_target_set = np.array(adv_input_target_set)
    print("shape of adv_input_set: ", np.shape(adv_input_set))
    print("shape of adv_input_target_set: ", np.shape(adv_input_target_set))
    adv_input_set = np.reshape(adv_input_set,(np.shape(adv_input_set)[0]*np.shape(adv_input_set)[1]*np.shape(adv_input_set)[2], np.shape(adv_input_set)[3], np.shape(adv_input_set)[4], np.shape(adv_input_set)[5]))
    adv_input_target_set = np.reshape(adv_input_target_set,(np.shape(adv_input_target_set)[0]*np.shape(adv_input_target_set)[1], np.shape(adv_input_target_set)[2], np.shape(adv_input_target_set)[3], np.shape(adv_input_target_set)[4]))

    print("generated adversarial training set")

    adv_set = cw.generate_np(adv_input_set, adv_input_target_set, **cw_params)


    x_train_aim = np.append(x_train, adv_input_set, axis = 0)
    x_train_app = np.append(x_train, adv_set, axis = 0)

    model_adv_trained = ModelBasicAE('model_adv_trained', nb_layers, nb_latent_size)
    recons_2 = model_adv_trained.get_layer(x, 'RECON')
    loss_2 = SquaredError(model_adv_trained) 
    train_ae(sess, loss_2, x_train_app, x_train_aim ,args=train_params, rng=rng, var_list=model_adv_trained.get_params())
    saver = tf.train.Saver()
    saver.save(sess, model_path)

    cw2 = CarliniWagnerAE(model_adv_trained,cl_model, sess=sess)

    adv_2 = cw2.generate_np(adv_inputs, adv_input_targets,
                       **cw_params)
    
    #print("shaep of adv: ", np.shape(adv))
    recon_orig = model_adv_trained.get_layer(adv_inputs, 'RECON')
    recon_adv = model_adv_trained.get_layer(adv_2, 'RECON')
    lat_orig = model_adv_trained.get_layer(x, 'LATENT')
    lat_orig_recon = model_adv_trained.get_layer(recons, 'LATENT')
    pred_adv_recon = cl_model.get_logits(recon_adv)

    #eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    eval_params = {'batch_size': 90}
    if targeted:
      noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recons, adv_inputs, adv_input_targets, adv_2, recon_adv,lat_orig, lat_orig_recon, args=eval_params)
      acc = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_target_y, adv_input_targets, args=eval_params_cls)
      print("noise: ", noise)
      #print("d1: ", d1)
      #print("d2: ", d2)
      #print("d1-d2: ", dist_diff)
      #print("Avg_dist_lat: ", avg_dist_lat)
      print("classifier acc: ", acc)

    recon_adv = sess.run(recon_adv)
    recon_orig = sess.run(recon_orig)
    #print("recon_adv[0]\n", recon_adv[0,:,:,0])
    curr_class = 0
    if viz_enabled:
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
                grid_viz_data_1[i,j] = adv_2[i*(nb_classes-1)+j-1]
              else:
                grid_viz_data[i,j] = recon_adv[i*(nb_classes-1) + j]
                grid_viz_data_1[i,j] = adv_2[i*(nb_classes-1)+j]


      #rint(grid_viz_data.shape)

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv_2 - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
      _ = grid_visual(grid_viz_data)
      _ = grid_visual(grid_viz_data_1)

    return report

#binarization defense
  if(binarization_defense == True or mean_filtering==True):

    #adv = sess.run(adv)
   # print(adv[0])
    if(binarization_defense==True):
      adv[adv>0.5] = 1.0
      adv[adv<=0.5] = 0.0
    else:
      #radius = 2
      #adv_list = [mean(adv[i,:,:,0], disk(radius)) for i in range(0, np.shape(adv)[0])]
      #adv = np.array(adv_list)
      #adv = np.expand_dims(adv, axis = 3)
      adv = uniform_filter(adv, 2)
      #adv = median_filter(adv, 2)
    #print("after bin ")
    #print(adv[0])

    recon_orig = model.get_layer(adv_inputs, 'RECON')
    recon_adv = model.get_layer(adv, 'RECON')
    lat_orig = model.get_layer(x, 'LATENT')
    lat_orig_recon = model.get_layer(recons, 'LATENT')
    pred_adv_recon = cl_model.get_logits(recon_adv)

    #eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    eval_params = {'batch_size': 90}
    if targeted:
      noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recons, adv_inputs, adv_input_targets, adv, recon_adv,lat_orig, lat_orig_recon, args=eval_params)
      acc1 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_target_y, adv_input_targets, args=eval_params_cls)
      acc2 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_input_y, adv_input_targets, args=eval_params_cls)
      print("d1: ", d1)
      print("d2: ", d2)
      print("noise: ", noise)
      print("classifier acc for target class: ", acc1)
      print("classifier acc for true class: ", acc2)

    recon_adv = sess.run(recon_adv)
    recon_orig = sess.run(recon_orig)
    #print("recon_adv[0]\n", recon_adv[0,:,:,0])
    curr_class = 0
    if viz_enabled:
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
      sess.close()

      _ = grid_visual(grid_viz_data)
      _ = grid_visual(grid_viz_data_1)

      #return report

def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  cifar10_cw_recon(viz_enabled=FLAGS.viz_enabled,
                    nb_epochs=FLAGS.nb_epochs,
                    batch_size=FLAGS.batch_size,
                    source_samples=FLAGS.source_samples,
                    learning_rate=FLAGS.learning_rate,
                    attack_iterations=FLAGS.attack_iterations,
                    model_path=FLAGS.model_path,
                    targeted=FLAGS.targeted, nb_filters = FLAGS.nb_filters)


if __name__ == '__main__':
  flags.DEFINE_boolean('viz_enabled', VIZ_ENABLED,
                       'Visualize adversarial ex.')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('source_samples', SOURCE_SAMPLES,
                       'Number of test inputs to attack')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('model_path', MODEL_PATH,
                      'Path to save or load the model file')
  flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS,
                       'Number of iterations to run attack; 1000 is good')
  flags.DEFINE_boolean('targeted', TARGETED,
                       'Run the tutorial in targeted mode?')
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')

  tf.app.run()
