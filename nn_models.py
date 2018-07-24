import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, Input, BatchNormalization
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from keras import initializers, regularizers, optimizers, losses

import h5py
import json

from nn_logging import getLogger
logger = getLogger()


# ______________________________________________________________________________
# New leaky relu
def NewLeakyReLU(x, alpha=0., max_value=None):
  return K.relu(x, alpha=alpha, max_value=max_value)

# ______________________________________________________________________________
# New tanh
def NewTanh(x):
  return K.tanh(x)
  #return 1.7159 * K.tanh(x * 2./3.)
  #return K.clip(x, -1., 1.)

# ______________________________________________________________________________
# Huber loss

def huber_loss(y_true, y_pred, delta=1.345):
  x = K.abs(y_true - y_pred)
  squared_loss = 0.5*K.square(x)
  absolute_loss = delta * (x - 0.5*delta)
  #xx = K.switch(x < delta, squared_loss, absolute_loss)
  xx = tf.where(x < delta, squared_loss, absolute_loss)  # needed for tensorflow
  return K.mean(xx, axis=-1)

def masked_huber_loss(y_true, y_pred, delta=1.345, mask_value=100.):
  x = K.abs(y_true - y_pred)
  squared_loss = 0.5*K.square(x)
  absolute_loss = delta * (x - 0.5*delta)
  #xx = K.switch(x < delta, squared_loss, absolute_loss)
  xx = tf.where(x < delta, squared_loss, absolute_loss)  # needed for tensorflow

  mask = K.not_equal(y_true, mask_value)
  mask = K.cast(mask, K.floatx())
  xx *= mask
  xx /= K.mean(mask)
  return K.mean(xx, axis=-1)

# ______________________________________________________________________________
# Binary crossentropy

# See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/losses.py
#      https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/backend.py
#def binary_crossentropy(y_true, y_pred):
#  return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def masked_binary_crossentropy(y_true, y_pred, mask_value=100.):
  xx = K.binary_crossentropy(y_true, y_pred)

  mask = K.not_equal(y_true, mask_value)
  mask = K.cast(mask, K.floatx())
  xx *= mask
  xx /= K.mean(mask)
  return K.mean(xx, axis=-1)

# ______________________________________________________________________________
# Callbacks

def lr_schedule(epoch, lr):
  if (epoch % 10) == 0 and epoch != 0:
    lr *= 0.95
  return lr

lr_decay = LearningRateScheduler(lr_schedule, verbose=0)

terminate_on_nan = TerminateOnNaN()

modelbestcheck = ModelCheckpoint(filepath='model_bchk.h5', monitor='val_loss', verbose=1, save_best_only=True)
modelbestcheck_weights = ModelCheckpoint(filepath='model_bchk_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)


# ______________________________________________________________________________
def create_model(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16, discr_loss_weight=1.0):
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000))(inputs)
  #x = Dropout(0.2)(x)
  x = Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000))(x)
  #x = Dropout(0.2)(x)
  x = Dense(nodes3, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000))(x)
  #x = Dropout(0.2)(x)

  regr = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='regr')(x)
  discr = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='discr')(x)

  # This creates a model that includes
  # the Input layer, three Dense layers and the Output layer
  model = Model(inputs=inputs, outputs=[regr, discr])

  # Set loss and optimizers
  #binary_crossentropy = losses.binary_crossentropy
  #mean_squared_error = losses.mean_squared_error

  adam = optimizers.Adam(lr=lr)
  model.compile(optimizer=adam,
    loss={'regr': masked_huber_loss, 'discr': masked_binary_crossentropy},
    loss_weights={'regr': 1.0, 'discr': discr_loss_weight},
    #metrics={'regr': ['acc', 'mse', 'mae'], 'discr': ['acc',]}
    )
  return model


# ______________________________________________________________________________
def create_model_sequential(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16):
  model = Sequential()
  model.add(Dense(nodes1, input_dim=nvariables, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000)))
  #model.add(Dropout(0.2))
  model.add(Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000)))
  #model.add(Dropout(0.2))
  model.add(Dense(nodes3, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000)))
  #model.add(Dropout(0.2))
  model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))

  adam = optimizers.Adam(lr=lr)
  model.compile(loss=huber_loss, optimizer=adam, metrics=['acc'])
  return model

def create_model_sequential_2layers(nvariables, lr=0.001, nodes1=64, nodes2=32):
  model = Sequential()
  model.add(Dense(nodes1, input_dim=nvariables, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000)))
  #model.add(Dropout(0.2))
  model.add(Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000)))
  #model.add(Dropout(0.2))
  model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))

  adam = optimizers.Adam(lr=lr)
  model.compile(loss=huber_loss, optimizer=adam, metrics=['acc'])
  return model

def create_model_sequential_1layer(nvariables, lr=0.001, nodes1=64):
  model = Sequential()
  model.add(Dense(nodes1, input_dim=nvariables, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0000)))
  #model.add(Dropout(0.2))
  model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))

  adam = optimizers.Adam(lr=lr)
  model.compile(loss=huber_loss, optimizer=adam, metrics=['acc'])
  return model


# ______________________________________________________________________________
def save_my_model(model, name='model'):
  # Store model to file
  model.summary()
  model.save(name + '.h5')
  model.save_weights(name + '_weights.h5')
  # Store model to json
  with open(name + '.json', 'w') as outfile:
    outfile.write(model.to_json())
  logger.info('Saved model as {0}.h5, {0}.json and {0}_weights.h5'.format(name))
  return

def load_my_model(name='model'):
  with open(name + '.json', 'r') as f:
    json_string = json.dumps(json.load(f))
    model = model_from_json(json_string)
  #model = load_model(name + '.h5')
  model.load_weights(name + '_weights.h5')
  logger.info('Loaded model from {0} and weights from {1}'.format(name + '.json', name + '_weights.h5'))
  return model


# ______________________________________________________________________________
# Scoring for cross-validation
# Based on https://github.com/keras-team/keras/blob/master/keras/wrappers/scikit_learn.py

from keras.wrappers.scikit_learn import KerasRegressor

class NewKerasRegressor(KerasRegressor):
  """KerasRegressor with custom 'score' function
  """

  def __init__(self, build_fn=None, reg_pt_scale=1.0, min_pt=20, max_pt=22, coverage=90., **sk_params):

    self.reg_pt_scale = reg_pt_scale
    self.min_pt = min_pt
    self.max_pt = max_pt
    self.coverage = coverage
    self.model = None

    super(KerasRegressor, self).__init__(build_fn=build_fn, **sk_params)

  def score(self, x, y, **kwargs):
    """Returns the mean loss on the given test data and labels.

    # Arguments
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y: array-like, shape `(n_samples,)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.evaluate`.

    # Returns
        score: float
            Mean accuracy of predictions on `x` wrt. `y`.
    """
    kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
    #loss = self.model.evaluate(x, y, **kwargs)

    # Prepare y_test_true, y_test_meas
    y_test_true = y
    if isinstance(y_test_true, list):
      y_test_true = y_test_true[0]
    y_test_true = y_test_true.copy()
    y_test_true = y_test_true.reshape(-1)
    y_test_true /= self.reg_pt_scale

    y_test_meas = self.model.predict(x, **kwargs)
    if isinstance(y_test_meas, list):
      y_test_meas = y_test_meas[0]
    y_test_meas = y_test_meas.reshape(-1)
    y_test_meas /= self.reg_pt_scale

    xx = np.abs(1.0/y_test_true)
    yy = np.abs(1.0/y_test_meas)

    reweight = lambda x, y, thres: 7.778 * np.power(x,-2.5) if y >= thres else 0.  # -2.5 instead of -3.5 because the parent distribution is already 1/pT-weighted

    xedges = [2., self.min_pt, self.max_pt, 42.]
    inds = np.digitize(xx, xedges[1:])

    xx_i = xx[inds==1]
    yy_i = yy[inds==1]
    pct = np.percentile(yy_i, [100-self.coverage], overwrite_input=True)

    thres = pct[0]
    yw = np.fromiter((reweight(xi, yi, thres) for (xi, yi) in zip(xx, yy)), xx.dtype)

    loss = np.sum(yw)

    #print "min_pt {0} max_pt {1} coverage {2} thres {3} loss {4}".format(self.min_pt, self.max_pt, self.coverage, thres, loss)

    if isinstance(loss, list):
      return -loss[0]
    return -loss
