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
import keras
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
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model

FLAGS = flags.FLAGS

VIZ_ENABLED = True
BATCH_SIZE = 90
NB_EPOCHS = 15
SOURCE_SAMPLES = 10
LEARNING_RATE = .002
CW_LEARNING_RATE = .4
ATTACK_ITERATIONS = 500  
MODEL_PATH = os.path.join('models', 'mnist_cw')
MODEL_PATH_CLS = os.path.join('models', 'mnist_cw_cl')
TARGETED = True
adv_train = False
binarization_defense = False
mean_filtering = True
NB_FILTERS = 4 #64
clean_train_ae = True
clean_train_cl = True

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
  
  '''
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
  '''
  '''
  def evaluate():
        do_eval(y_logits, x_test, y_test, 'clean_train_clean_eval', False)

  filepath_ae = "clean_model_cifar10_ae.joblib"
  filepath_cl = "classifier_cifar10.joblib"

  
# Define TF model graph
  model = ModelBasicAE('model', nb_layers, nb_latent_size)
  #cl_model = ModelCls('cl_model')
  #cl_model = ModelAllConvolutional('model1', nb_classes, nb_filters,
  #                                input_shape=[32, 32, 3])
  #preds = model.get_logits(x)
  recons = model.get_layer(x, 'RECON')
  latent1_orig = model.get_layer(x, 'LATENT')
  latent1_orig_recon = model.get_layer(recons, 'LATENT')

  loss = SquaredError(model)
  print("Defined TensorFlow model graph.")
  #y_logits = cl_model.get_logits(x)
  #loss_cls = CrossEntropy(cl_model, smoothing=label_smoothing)
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
      'nb_epochs': 4,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  
  rng = np.random.RandomState([2017, 8, 30])
  # check if we've trained before, and if we have, use that pre-trained model
  #if os.path.exists(model_path + ".meta"):
   # tf_model_load(sess, model_path)
  #else:
  #eval_params_cls = {'batch_size': batch_size}
  # Evaluate the accuracy of the MNIST model on legitimate test examples
  eval_params = {'batch_size': batch_size}
  
  def do_eval(recons, x_orig, x_target, y_orig, y_target, report_key, is_adv=False, x_adv = None, recon_adv = False, lat_orig = None, lat_orig_recon = None):
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

  def evaluate_ae():
    do_eval(recons, x_test, x_test, y_test, y_test, 'clean_train_clean_eval', False, None, None, latent1_orig, latent1_orig_recon)

  print("Training autoencoder")
  train_ae(sess, loss, x_train,x_train, evaluate = evaluate_ae, args=train_params, rng=rng, var_list=model.get_params())
  #with sess.as_default():
   # save(filepath_ae, model)
  '''
  save_dir= 'models'
  model_name = 'cifar10_AE'
  model_path_ae = os.path.join(save_dir, model_name)

  if clean_train_ae==True:
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

    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    #es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    #chkpt = saveDir + 'AutoEncoder_Cifar10_Deep_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    #cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model.fit(x_train, x_train,
                    batch_size=128,
                    epochs=2,
                    verbose=1,
                    validation_data=(x_test, x_test),
                    #callbacks=[es_cb, cp_cb],
                    shuffle=True)
    score = model.evaluate(x_test, x_test, verbose=1)
    print(score)
    model.save(model_path_ae)
    print('Saved trained model at %s ' % model_path)

  else:
    model = load_model(model_path_ae)

  num_classes = 10
  save_dir= 'models'
  model_name = 'cifar10_CNN'
  model_path_cls = os.path.join(save_dir, model_name)

  if clean_train_cl == True:
    print("Training CNN classifier")
    cl_model = Sequential()
    cl_model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    cl_model.add(Activation('relu'))
    cl_model.add(Conv2D(32, (3, 3)))
    cl_model.add(Activation('relu'))
    cl_model.add(MaxPooling2D(pool_size=(2, 2)))
    cl_model.add(Dropout(0.25))

    cl_model.add(Conv2D(64, (3, 3), padding='same'))
    cl_model.add(Activation('relu'))
    cl_model.add(Conv2D(64, (3, 3)))
    cl_model.add(Activation('relu'))
    cl_model.add(MaxPooling2D(pool_size=(2, 2)))
    cl_model.add(Dropout(0.25))

    cl_model.add(Flatten())
    cl_model.add(Dense(512))
    cl_model.add(Activation('relu'))
    cl_model.add(Dropout(0.5))
    cl_model.add(Dense(num_classes))
    cl_model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    cl_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    cl_model.fit(x_train, y_train,
              batch_size=90,
              epochs= 4,
              validation_data=(x_test, y_test),
              shuffle=True)
    
    cl_model.save(model_path_cls)
    print('Saved trained model at %s ' % model_path)

  else:
    cl_model = load_model(model_path_cls)

    # Score trained model.
  scores = cl_model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])

  '''
  train(sess, loss_cls, None, None,
            dataset_train=dataset_train, dataset_size=dataset_size,
            evaluate=eval_cls, args=train_params_cls, rng=rng,
            var_list=cl_model.get_params())
  '''
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

    

  adv_ys = np.array([one_hot] * source_samples,
                      dtype=np.float32).reshape((source_samples *
                                                 nb_classes, nb_classes))
  yname = "y_target"

  cw_params_batch_size = source_samples * (nb_classes-1)
  
  cw_params = {'binary_search_steps': 4,
               yname: adv_ys,
               'max_iterations': attack_iterations,
               'learning_rate': CW_LEARNING_RATE,
               'batch_size': cw_params_batch_size,
               'initial_const': 1}

  adv = cw.generate_np(adv_inputs, adv_input_targets,
                       **cw_params)
  adv = sess.run(adv)
  #print("shaep of adv: ", np.shape(adv))
  '''
  recons = model.get_layer(x, 'RECON')
  recon_orig = model.get_layer(adv_inputs, 'RECON')
  recon_adv = model.get_layer(adv, 'RECON')
  lat_orig = model.get_layer(x, 'LATENT')
  lat_orig_recon = model.get_layer(recons, 'LATENT')
  #pred_adv_recon = cl_model.get_logits(recon_adv)
  '''
  recon_orig = model.predict(adv_inputs)
  recon_adv = model.predict(adv)
  #eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
  #eval_params = {'batch_size': 90}
   
  #acc_1 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_target_y, adv_input_targets, args=eval_params_cls)
  #acc_2 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_input_y, adv_input_targets, args=eval_params_cls)
  
  #noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recons, adv_inputs, adv_input_targets, adv, recon_adv,lat_orig, lat_orig_recon, args=eval_params)
  shape = np.shape(adv_inputs)
  noise = reduce_sum(np.square(adv_inputs - adv), list(range(1, len(shape))))
  print("noise: ", noise)
  #recon_adv = sess.run(recon_adv)
  #recon_orig = sess.run(recon_orig)
  scores1 = cl_model.evaluate(recon_adv, adv_input_y, verbose=1)
  scores2 = cl_model.evaluate(recon_adv, adv_target_y, verbose = 1)
  print("classifier acc_target: ", scores2[1])
  print("classifier acc_true: ", scores1[1])

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
    #_ = grid_visual(grid_viz_data)
    #_ = grid_visual(grid_viz_data_1)

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
    plt.savefig('cifar10_fig1')
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')
    for yy in range(num_rows):
      for xx in range(num_cols):
        figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy * num_cols))
        plt.axis('off')
        plt.imshow(grid_viz_data_1[xx, yy, :, :, :])

    # Draw the plot and return
    plt.savefig('cifar10_fig2')

  
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

    model_name = 'cifar10_AE_adv'
    model_path_ae = os.path.join(save_dir, model_name)

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

    model2 = Model(input_img, decoded)
    model2.compile(optimizer='adam', loss='binary_crossentropy')

    model2.fit(x_train_app, x_train_aim,
                    batch_size=128,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, x_test),
                    callbacks=[es_cb, cp_cb],
                    shuffle=True)
    score = model.evaluate(x_test, x_test, verbose=1)
    print(score)
    model2.save(model_path_ae_adv)
    print('Saved adv trained model at %s ' % model_path)

    '''
    model_adv_trained = ModelBasicAE('model_adv_trained', nb_layers, nb_latent_size)
    recons_2 = model_adv_trained.get_layer(x, 'RECON')
    loss_2 = SquaredError(model_adv_trained) 
    train_ae(sess, loss_2, x_train_app, x_train_aim ,args=train_params, rng=rng, var_list=model_adv_trained.get_params())
    saver = tf.train.Saver()
    saver.save(sess, model_path)
    '''

    cw2 = CarliniWagnerAE(model_adv_trained,cl_model, sess=sess)

    adv_2 = cw2.generate_np(adv_inputs, adv_input_targets,
                       **cw_params)
    
    recon_adv= model2.predict(adv)
    recon_orig = model2.predict(adv_inputs)
    #print("shaep of adv: ", np.shape(adv))
    '''
    recon_orig = model_adv_trained.get_layer(adv_inputs, 'RECON')
    recon_adv = model_adv_trained.get_layer(adv_2, 'RECON')
    lat_orig = model_adv_trained.get_layer(x, 'LATENT')
    lat_orig_recon = model_adv_trained.get_layer(recons, 'LATENT')
    '''
    #pred_adv_recon = cl_model.get_logits(recon_adv)

    #eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    #eval_params = {'batch_size': 90}
    if targeted:
      #noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recons, adv_inputs, adv_input_targets, adv_2, recon_adv,lat_orig, lat_orig_recon, args=eval_params)
      #acc = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_target_y, adv_input_targets, args=eval_params_cls)
      noise = reduce_sum(tf.square(adv_inputs - adv_2), list(range(1, len(shape))))
      print("noise: ", noise)
      #print("d1: ", d1)
      #print("d2: ", d2)
      #print("d1-d2: ", dist_diff)
      #print("Avg_dist_lat: ", avg_dist_lat)
      #print("classifier acc: ", acc)
    '''  
    recon_adv = sess.run(recon_adv)
    recon_orig = sess.run(recon_orig)
    '''
    scores1 = cl_model.evaluate(recon_adv, adv_input_y, verbose=1)
    scores2 = cl_model.eval_params(recon_adv, adv_target_y, verbose = 1)
    print("classifier acc_target: ", scores2[1])
    print("classifier acc_true: ", scores1[1])

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
      #_ = grid_visual(grid_viz_data)
      #_ = grid_visual(grid_viz_data_1)
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

          if num_channels == 1:
            plt.imshow(grid_viz_data[xx, yy, :, :, 0])
          else:
            plt.imshow(grid_viz_data[xx, yy, :, :, :])

      # Draw the plot and return
      plt.savefig('cifar10_fig1_adv_trained')
      figure = plt.figure()
      figure.canvas.set_window_title('Cleverhans: Grid Visualization')
      for yy in range(num_rows):
        for xx in range(num_cols):
          figure.add_subplot(num_rows, num_cols, (xx + 1) + (yy * num_cols))
          plt.axis('off')
          plt.imshow(grid_viz_data_1[xx, yy, :, :, :])

      # Draw the plot and return
      plt.savefig('cifar10_fig2_adv_trained')

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
    '''
    recons = model.get_layer(x, 'RECON')
    recon_orig = model.get_layer(adv_inputs, 'RECON')
    recon_adv = model.get_layer(adv, 'RECON')
    lat_orig = model.get_layer(x, 'LATENT')
    lat_orig_recon = model.get_layer(recon_orig, 'LATENT')
    '''
    recon_orig = model.predict(adv_inputs)
    recon_adv = model.predict(adv)

    #pred_adv_recon = cl_model.get_logits(recon_adv)

    #eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    eval_params = {'batch_size': 90}
    if targeted:
      #noise, d1, d2, dist_diff, avg_dist_lat = model_eval_ae(sess, x, x_t,recons, adv_inputs, adv_input_targets, adv, recon_adv,lat_orig, lat_orig_recon, args=eval_params)
      #acc1 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_target_y, adv_input_targets, args=eval_params_cls)
      #acc2 = model_eval(sess, x, y, pred_adv_recon, x_t, adv_inputs, adv_input_y, adv_input_targets, args=eval_params_cls)
      
      #print("d1: ", d1)
      #print("d2: ", d2)
      noise = reduce_sum(tf.square(x_orig - x_adv), list(range(1, len(shape))))
      print("noise: ", noise)
      #print("classifier acc for target class: ", acc1)
      #print("classifier acc for true class: ", acc2)
    '''
    recon_adv = sess.run(recon_adv)
    recon_orig = sess.run(recon_orig)
    '''
    scores1 = cl_model.evaluate(recon_adv, adv_input_y, verbose=1)
    scores2 = cl_model.evalluate(recon_adv, adv_target_y, verbose = 1)
    print("classifier acc_target: ", scores2[1])
    print("classifier acc_true: ", scores1[1])
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

      #_ = grid_visual(grid_viz_data)
      #_ = grid_visual(grid_viz_data_1)
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
    plt.savefig('cifar10_fig1_bin')
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
    plt.savefig('cifar10_fig2_bin')

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
