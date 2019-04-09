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
from cleverhans.dataset import CIFAR10
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.serial import save
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cleverhans.utils_keras import cnn_model, ae_model, cnn_cl_model
from keras.optimizers import Adam
from keras.models import Model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
from keras.preprocessing.image import ImageDataGenerator
from cleverhans.train import train
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
mean_filtering = True
clean_train_ae = False
clean_train_cl = False  
train_further = False
targeted = True
SOURCE_SAMPLES = 10
viz_enabled = True
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
  rng = np.random.RandomState()

  source_samples = 10
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
  #z = tf.placeholder(tf.float32, shape = (None, nb_latent_size))
  #z_t = tf.placeholder(tf.float32, shape = (None, nb_latent_size))
  '''
  save_dir= 'models'
  model_name = 'cifar10_AE.h5'
  model_path_ae = os.path.join(save_dir, model_name)
  '''
  model_ae= ae_model(x, img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels)
  recon = model_ae(x)
  #print("recon: ",recon)
  print("Defined TensorFlow model graph.")

  def evaluate_ae():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 128}
    noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recon, x_train, x_train, args=eval_params)
    print("reconstruction distance: ", d1)
  
  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      #'train_dir': train_dir_ae,
      #'filename': filename
  }
  rng = np.random.RandomState([2017, 8, 30])
  #if not os.path.exists(train_dir_ae):
   # os.mkdir(train_dir_ae)

  #ckpt = tf.train.get_checkpoint_state(train_dir_ae)
  #print(train_dir_ae, ckpt)
  #ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
  wrap_ae = KerasModelWrapper(model_ae)

  if clean_train_ae==True:
    print("Training AE")
    loss = SquaredError(wrap_ae)
    train_ae(sess, loss, x_train, x_train, evaluate=evaluate_ae,
          args=train_params, rng=rng)
    saver = tf.train.Saver()
    saver.save(sess, "train_dir/model_ae.ckpt")
    print("saved model")
    

  else:
    print("Loading AE")
    saver = tf.train.Saver()
    #print(ckpt_path)
    saver.restore(sess, "train_dir/model_ae.ckpt")
    evaluate_ae()
    if train_further:
      train_params = {
        'nb_epochs': 100,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        #'train_dir': train_dir_ae,
        #'filename': filename
      }
      #training with the saved model as starting point
      loss = SquaredError(wrap_ae)
      train_ae(sess, loss, x_train, x_train, evaluate=evaluate_ae,
            args=train_params, rng=rng)
      saver = tf.train.Saver()
      saver.save(sess, "train_dir/model_ae_final.ckpt")
      evaluate_ae()
      print("Model loaded and trained for more epochs")
    

  num_classes = 10
  '''
  save_dir= 'models'
  model_name = 'cifar10_CNN.h5'
  model_path_cls = os.path.join(save_dir, model_name)
  '''
  cl_model = cnn_cl_model(img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels, nb_filters=64,
                    nb_classes=nb_classes)
  preds_cl = cl_model(x)
  def do_eval_cls(preds, x_set, y_set, x_tar_set,report_key, is_adv = None):
    acc = model_eval(sess, x, y, preds, x_t, x_set, y_set, x_tar_set, args=eval_params_cls)

  def evaluate():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_cl,x_t, x_test, y_test, x_test,args=eval_params)
    report.clean_train_clean_eval = acc
#        assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate examples: %0.4f' % acc)

  train_params = {
      'nb_epochs': 100,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      #'train_dir': train_dir_cl,
      #'filename': filename
  }
  rng = np.random.RandomState([2017, 8, 30])
  #if not os.path.exists(train_dir_cl):
   # os.mkdir(train_dir_cl)

  #ckpt = tf.train.get_checkpoint_state(train_dir_cl)
  #print(train_dir_cl, ckpt)
  #ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
  wrap_cl = KerasModelWrapper(cl_model)

  if clean_train_cl == True:  
    train_params = {
        'nb_epochs': 50,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        #'train_dir': train_dir_cl,
        #'filename': filename
      }
    print("Training CNN Classifier")
    '''
    datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
    datagen.fit(x_train)
    '''
    loss_cl = CrossEntropy(wrap_cl, smoothing=label_smoothing)
    #for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size = 128):
     # train(sess, loss_cl, x_batch, y_batch, tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5), evaluate=evaluate,
      #          args=train_params, rng=rng)
    train(sess, loss_cl, x_train, y_train, evaluate=evaluate, optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001, decay = 1e-6),
          args=train_params, rng=rng)
    saver = tf.train.Saver()
    saver.save(sess, "train_dir/model_cnn_cl.ckpt")
    print("saved model at ", "train_dir/model_cnn_cl.ckpt")
    
  else:
    print("Loading CNN Classifier")
    saver = tf.train.Saver()
    #print(ckpt_path)
    saver.restore(sess, "train_dir/model_cnn_cl.ckpt")
    evaluate()
    if(train_further):
      train_params = {
        'nb_epochs': 10,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': train_dir_cl,
        'filename': filename
      }
      loss_cl = CrossEntropy(wrap_cl, smoothing=label_smoothing)
      train(sess, loss_cl, x_train, y_train, evaluate=evaluate, optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001, decay = 1e-6),
            args=train_params, rng=rng)
      saver = tf.train.Saver()
      saver.save(sess, "train_dir/model_cl_hi_acc.ckpt")
      print("Model loaded and trained further")
      evaluate()



    # Score trained model.
  '''
  scores = cl_model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])
  cl_model_wrap = KerasModelWrapper(cl_model)
` '''
  ###########################################################################
  # Craft adversarial examples using Carlini and Wagner's approach
  ###########################################################################
  nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
  print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
        ' adversarial examples')
  print("This could take some time ...")

  # Instantiate a CW attack object
 #cw = CarliniWagnerAE(wrap_ae,wrap_cl, sess=sess)

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

    

  adv_ys = np.array([one_hot] * source_samples,
                      dtype=np.float32).reshape((source_samples *
                                                 nb_classes, nb_classes))
  yname = "y_target"

  fgsm_params = {
      'eps': 0.3,
      'clip_min': 0.,
      'clip_max': 1.
  }

  fgsm = FastGradientMethodAe(wrap_ae, sess=sess)
  adv = fgsm.generate(x,x_t, **fgsm_params)

  adv = sess.run(adv, {x: adv_inputs, x_t: adv_input_targets})

  recon_orig = wrap_ae.get_layer(x, 'activation_7')
  recon_orig = sess.run(recon_orig, feed_dict = {x: adv_inputs})
  recon_adv = wrap_ae.get_layer(x, 'activation_7')
  recon_adv = sess.run(recon_adv, feed_dict = {x: adv})
  pred_adv_recon = wrap_cl.get_logits(x)
  pred_adv_recon = sess.run(pred_adv_recon, {x:recon_adv})

  #scores1 = cl_model.evaluate(recon_adv, adv_input_y, verbose=1)
  #scores2 = cl_model.evaluate(recon_adv, adv_target_y, verbose = 1)
  #acc_1 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_target_y, adv_input_targets, args=eval_params_cls)
  #acc_2 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_input_y, adv_input_targets, args=eval_params_cls)
  shape = np.shape(adv_inputs)
  noise = np.sum(np.square(adv-adv_inputs))/(np.shape(adv)[0])
  noise = pow(noise,0.5)
  d1 = np.sum(np.square(recon_adv-adv_inputs))/(np.shape(adv_inputs)[0])
  d2 = np.sum(np.square(recon_adv-adv_input_targets))/(np.shape(adv_inputs)[0])
  acc_1 = (sum(np.argmax(pred_adv_recon, axis=-1)==
                             np.argmax(adv_target_y, axis=-1)))/(np.shape(adv_target_y)[0])
  acc_2 = (sum(np.argmax(pred_adv_recon, axis=-1)==
                             np.argmax(adv_input_y, axis=-1)))/(np.shape(adv_target_y)[0])
  print("noise: ", noise)
  print("d1: ", d1)
  print("d2: ", d2)
  print("classifier acc_target: ", acc_1)
  print("classifier acc_true: ", acc_2)

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
  # Finally, block & display a grid of all the adversarial examples
  
  if viz_enabled:
    
    plt.ioff()
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')

    # Add the images to the plot
    num_cols = grid_viz_data.shape[0]
    num_rows = grid_viz_data.shape[1]
    num_channels = grid_viz_data.shape[4]
    for yy in range(num_rows):
      for xx in range(num_cols):
        figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy * num_cols))
        plt.axis('off')
        plt.imshow(grid_viz_data[xx, yy, :, :, :])

    # Draw the plot and return
    plt.savefig('cifar10_fgsm_fig1')
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')
    for yy in range(num_rows):
      for xx in range(num_cols):
        figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy * num_cols))
        plt.axis('off')
        plt.imshow(grid_viz_data_1[xx, yy, :, :, :])

    # Draw the plot and return
    plt.savefig('cifar10_fgsm_fig2')

  
  #return report
  if binarization:

    adv[adv>0.5] = 1.0
    adv[adv<=0.5] = 0.0
    
     
    recon_orig = wrap_ae.get_layer(x, 'activation_7')
    recon_adv = wrap_ae.get_layer(x, 'activation_7')
    #pred_adv = wrap_cl.get_logits(x)
    recon_orig = sess.run(recon_orig, {x: adv_inputs})
    recon_adv = sess.run(recon_adv, {x: adv})
    #pred_adv = sess.run(pred_adv, {x: recon_adv})
    pred_adv_recon = wrap_cl.get_logits(x)
    pred_adv_recon = sess.run(pred_adv_recon, {x:recon_adv})

    eval_params = {'batch_size': 90}
    if targeted:
     
      noise = np.sum(np.square(adv-adv_inputs))/(np.shape(adv)[0])
      noise = pow(noise,0.5)
      d1 = np.sum(np.square(recon_adv-adv_inputs))/(np.shape(adv_inputs)[0])
      d2 = np.sum(np.square(recon_adv-adv_input_targets))/(np.shape(adv_inputs)[0])
      acc_1 = (sum(np.argmax(pred_adv_recon, axis=-1)==
                               np.argmax(adv_target_y, axis=-1)))/(np.shape(adv_target_y)[0])
      acc_2 = (sum(np.argmax(pred_adv_recon, axis=-1)==
                               np.argmax(adv_input_y, axis=-1)))/(np.shape(adv_target_y)[0])
      print("noise: ", noise)
      print("d1: ", d1)
      print("d2: ", d2)
      print("classifier acc_target: ", acc_1)
      print("classifier acc_true: ", acc_2)


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
      

    plt.ioff()
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')

    # Add the images to the plot
    num_cols = grid_viz_data.shape[0]
    num_rows = grid_viz_data.shape[1]
    num_channels = grid_viz_data.shape[4]
    for yy in range(num_rows):
      for xx in range(num_cols):
        figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy* num_cols))
        plt.axis('off')

        if num_channels == 1:
          plt.imshow(grid_viz_data[xx, yy, :, :, 0])
        else:
          plt.imshow(grid_viz_data[xx, yy, :, :, :])

    # Draw the plot and return
    plt.savefig('cifar10_fgsm_fig1_bin')
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')
    for yy in range(num_rows):
      for xx in range(num_cols):
        figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy * num_cols))
        plt.axis('off')

        if num_channels == 1:
          plt.imshow(grid_viz_data_1[xx, yy, :, :, 0])
        else:
          plt.imshow(grid_viz_data_1[xx, yy, :, :, :])

    # Draw the plot and return
    plt.savefig('cifar10_fgsm_fig2_bin')

  if(mean_filtering ==True):

      adv = uniform_filter(adv, 2)

      recon_orig = wrap_ae.get_layer(x, 'activation_7')
      recon_adv = wrap_ae.get_layer(x, 'activation_7')
      pred_adv_recon = wrap_cl.get_logits(x)
      recon_orig = sess.run(recon_orig, {x: adv_inputs})
      recon_adv = sess.run(recon_adv, {x: adv})
      pred_adv_recon = sess.run(pred_adv_recon, {x: recon_adv})

      eval_params = {'batch_size': 90}
      
      noise = np.sum(np.square(adv-adv_inputs))/(np.shape(adv)[0])
      noise = pow(noise,0.5)
      d1 = np.sum(np.square(recon_adv-adv_inputs))/(np.shape(adv_inputs)[0])
      d2 = np.sum(np.square(recon_adv-adv_input_targets))/(np.shape(adv_inputs)[0])
      acc_1 = (sum(np.argmax(pred_adv_recon, axis=-1)==
                               np.argmax(adv_target_y, axis=-1)))/(np.shape(adv_target_y)[0])
      acc_2 = (sum(np.argmax(pred_adv_recon, axis=-1)==
                               np.argmax(adv_input_y, axis=-1)))/(np.shape(adv_target_y)[0])
      print("noise: ", noise)
      print("d1: ", d1)
      print("d2: ", d2)
      print("classifier acc_target: ", acc_1)
      print("classifier acc_true: ", acc_2)


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
        

      plt.ioff()
      figure = plt.figure()
      figure.canvas.set_window_title('Cleverhans: Grid Visualization')

      # Add the images to the plot
      num_cols = grid_viz_data.shape[0]
      num_rows = grid_viz_data.shape[1]
      num_channels = grid_viz_data.shape[4]
      for yy in range(num_rows):
        for xx in range(num_cols):
          figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy* num_cols))
          plt.axis('off')

          if num_channels == 1:
            plt.imshow(grid_viz_data[xx, yy, :, :, 0])
          else:
            plt.imshow(grid_viz_data[xx, yy, :, :, :])

      # Draw the plot and return
      plt.savefig('cifar10_fgsm_fig1_mean')
      figure = plt.figure()
      figure.canvas.set_window_title('Cleverhans: Grid Visualization')
      for yy in range(num_rows):
        for xx in range(num_cols):
          figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy * num_cols))
          plt.axis('off')

          if num_channels == 1:
            plt.imshow(grid_viz_data_1[xx, yy, :, :, 0])
          else:
            plt.imshow(grid_viz_data_1[xx, yy, :, :, :])

      # Draw the plot and return
      plt.savefig('cifar10_fgsm_fig2_mean')

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