import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model, clone_model, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, Input, Concatenate, BatchNormalization
from keras.callbacks import LearningRateScheduler, TerminateOnNaN, ModelCheckpoint
from keras.regularizers import Regularizer
from keras.constraints import Constraint
from keras import initializers, regularizers, optimizers, losses

import h5py
import json
import functools


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
# New elu
def NewElu(x, alpha=1.0):
  return K.elu(x, alpha) + alpha*1.0 + 1e-15

# ______________________________________________________________________________
class LCountParams(Regularizer):
  """Regularizer that penalizes large number of parameters.
  Copied from class L1L2 from https://github.com/keras-team/keras/blob/master/keras/regularizers.py

  # Arguments
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.
  """

  def __init__(self, l1=0., l2=0.):
    self.l1 = K.cast_to_floatx(l1)
    self.l2 = K.cast_to_floatx(l2)

  def __call__(self, x):
    regularization = 0.
    if self.l1:
      regularization += self.l1 * K.abs(K.cast_to_floatx(K.count_params(x)))
    if self.l2:
      regularization += self.l2 * K.square(K.cast_to_floatx(K.count_params(x)))
    return regularization

  def get_config(self):
    return {'l1': float(self.l1),
            'l2': float(self.l2)}

# ______________________________________________________________________________
class ZeroSomeWeights(Constraint):
  """ZeroSomeWeights weight constraint.
  Constrains certain weights incident to each hidden unit
  to be zero.
  Copied from https://github.com/hls-fpga-machine-learning/keras-training/blob/muon/models/constraints.py

  # Arguments
      binary_tensor: binary tensor of 0 or 1s corresponding to which weights to zero.
  """

  def __init__(self, binary_tensor=None):
    self.binary_tensor = binary_tensor

  def __call__(self, w):
    if self.binary_tensor is not None:
      w *= K.cast(self.binary_tensor, K.floatx())
    return w

  def get_config(self):
    return {'binary_tensor': self.binary_tensor.tolist()}

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
# Custom objects

def update_keras_custom_objects():
  custom_objects = {
    'masked_huber_loss': masked_huber_loss,
    'masked_binary_crossentropy': masked_binary_crossentropy,
    'NewLeakyReLU': NewLeakyReLU,
    'NewTanh': NewTanh,
    'NewElu': NewElu,
    'LCountParams': LCountParams,
    'ZeroSomeWeights': ZeroSomeWeights,
  }

  from keras.utils.generic_utils import get_custom_objects
  get_custom_objects().update(custom_objects)

# ______________________________________________________________________________
# Create models
def create_model(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16, discr_loss_weight=1.0, l1_reg=0.0, l2_reg=0.0):
  regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer)(inputs)
  #x = Dropout(0.2)(x)
  if nodes2:
    x = Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer)(x)
    #x = Dropout(0.2)(x)
    if nodes3:
      x = Dense(nodes3, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer)(x)
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
  model.summary()
  return model

# ______________________________________________________________________________
def create_model_bn(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16, discr_loss_weight=1.0, l1_reg=0.0, l2_reg=0.0, use_bn=True):
  regularizer = regularizers.L1L2(l1=l1_reg, l2=l2_reg)
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False)(inputs)
  if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
  x = Activation('tanh')(x)
  if nodes2:
    x = Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False)(x)
    if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
    x = Activation('tanh')(x)
    if nodes3:
      x = Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, use_bias=False)(x)
      if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
      x = Activation('tanh')(x)

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
  model.summary()
  return model

# ______________________________________________________________________________
def create_model_pruned(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16, discr_loss_weight=1.0,
                        l1_reg=0.0, l2_reg=0.0, use_bn=True,
                        constraint1=None, constraint2=None, constraint3=None):
  regularizer = None  # disable
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint1, use_bias=False)(inputs)
  if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
  x = Activation('tanh')(x)
  if nodes2:
    x = Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint2, use_bias=False)(x)
    if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
    x = Activation('tanh')(x)
    if nodes3:
      x = Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint3, use_bias=False)(x)
      if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
      x = Activation('tanh')(x)

  regr = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='regr')(x)
  discr = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='discr')(x)

  # Create model
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
  model.summary()
  return model

# ______________________________________________________________________________
def log_prob_normal(x, loc=0.0, scale=1.0):
  y = (x - loc) / scale
  y = -0.5*K.square(y) - 0.5*K.log(2*np.pi) - K.log(scale)
  return y

def log_prob_softmax(x):
  return tf.nn.log_softmax(K.clip(x,K.epsilon(),1.0))

def log_prob(x, mus, sigmas, pi):
  distribution_log_probs = log_prob_normal(x, mus, sigmas)
  #cat_log_probs = log_prob_softmax(pi)
  cat_log_probs = K.log(pi)
  final_log_probs = tf.add(distribution_log_probs, cat_log_probs)
  result = tf.reduce_logsumexp(final_log_probs, axis=-1)
  return result

def mixture_loss(y_true, y_pred, mus, sigmas, pi):
  loss = -log_prob(y_true, mus, sigmas, pi)
  return K.mean(loss, axis=-1)

def mixture_loss_for_keras(mus, sigmas, pi):
  @functools.wraps(mixture_loss)
  def loss(y_true, y_pred):
    return mixture_loss(y_true, y_pred, mus, sigmas, pi)
  return loss

def create_model_mdn(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16, mixture=16, discr_loss_weight=1.0,
                     l1_reg=0.0, l2_reg=0.0, use_bn=True,
                     constraint1=None, constraint2=None, constraint3=None):
  regularizer = None  # disable
  inputs = Input(shape=(nvariables,), dtype='float32')

  x = Dense(nodes1, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint1, use_bias=False)(inputs)
  if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
  x = Activation('tanh')(x)
  if nodes2:
    x = Dense(nodes2, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint2, use_bias=False)(x)
    if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
    x = Activation('tanh')(x)
    if nodes3:
      x = Dense(nodes3, kernel_initializer='glorot_uniform', kernel_regularizer=regularizer, kernel_constraint=constraint3, use_bias=False)(x)
      if use_bn: x = BatchNormalization(center=True, scale=True, epsilon=1e-4, momentum=0.9)(x)
      x = Activation('tanh')(x)

  mus = Dense(mixture, activation=None, kernel_initializer='glorot_uniform', name='mus')(x)  # the means
  sigmas = Dense(mixture, activation=NewElu, kernel_initializer='glorot_uniform', name='sigmas')(x)  # the variance
  pi = Dense(mixture, activation='softmax', kernel_initializer='glorot_uniform', name='pi')(x)  # the mixture components

  outputs = Concatenate(axis=1)([pi, mus, sigmas])

  # Create model
  model = Model(inputs=inputs, outputs=outputs)

  # Set loss and optimizers
  adam = optimizers.Adam(lr=lr)
  keras_loss = mixture_loss_for_keras(mus=mus, sigmas=sigmas, pi=pi)
  model.compile(optimizer=adam, loss=keras_loss)
  model.summary()
  return model


# ______________________________________________________________________________
def create_model_sequential(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16, l1_reg=0.0, l2_reg=0.0):
  regularizer = regularizers.L1L2(l1=l1_reg, l2=l2_reg)

  model = Sequential()
  model.add(Dense(nodes1, input_dim=nvariables, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
  #model.add(Dropout(0.2))
  if nodes2:
    model.add(Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
    #model.add(Dropout(0.2))
    if nodes3:
      model.add(Dense(nodes3, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
      #model.add(Dropout(0.2))

  model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))

  adam = optimizers.Adam(lr=lr)
  model.compile(loss=huber_loss, optimizer=adam, metrics=['acc'])
  model.summary()
  return model

# ______________________________________________________________________________
def create_model_sequential_regularized(nvariables, lr=0.001, nodes1=64, nodes2=32, nodes3=16, l1_reg=0.0, l2_reg=0.0):
  #regularizer = regularizers.L1L2(l1=l1_reg, l2=l2_reg)
  regularizer = LCountParams(l1=l1_reg, l2=l2_reg)

  model = Sequential()
  model.add(Dense(nodes1, input_dim=nvariables, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
  #model.add(Dropout(0.2))
  if nodes2:
    model.add(Dense(nodes2, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
    #model.add(Dropout(0.2))
    if nodes3:
      model.add(Dense(nodes3, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizer))
      #model.add(Dropout(0.2))

  model.add(Dense(1, activation='linear', kernel_initializer='glorot_uniform'))

  adam = optimizers.Adam(lr=lr)
  model.compile(loss=huber_loss, optimizer=adam, metrics=['acc'])
  model.summary()
  return model

# ______________________________________________________________________________
# Save/Load models
def save_my_model(model, name='model'):
  # Store model to file
  #model.summary()
  model.save(name + '.h5')
  model.save_weights(name + '_weights.h5')
  # Store model to json
  with open(name + '.json', 'w') as outfile:
    outfile.write(model.to_json())
  return

def load_my_model(name='model', weights_name='model_weights'):
  with open(name + '.json', 'r') as f:
    json_string = json.dumps(json.load(f))
    json_string = json_string.replace('"axis": [3]', '"axis": 3')  # dirty hack
    json_string = json_string.replace('"axis": [2]', '"axis": 2')  # dirty hack
    json_string = json_string.replace('"axis": [1]', '"axis": 1')  # dirty hack
    model = model_from_json(json_string)
  #model = load_model(name + '.h5')
  model.load_weights(weights_name + '.h5')
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

  def score2(self, x, y, **kwargs):
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

    reweight = lambda x, y, thresh: 7.778 * np.power(x,-2.5) if y >= thresh else 0.  # -2.5 instead of -3.5 because the parent distribution is already 1/pT-weighted

    xedges = [2., self.min_pt, self.max_pt, 42.]
    inds = np.digitize(xx, xedges[1:])

    xx_i = xx[inds==1]
    yy_i = yy[inds==1]
    pct = np.percentile(yy_i, [100-self.coverage], overwrite_input=True)

    thresh = pct[0]
    yw = np.fromiter((reweight(xi, yi, thresh) for (xi, yi) in zip(xx, yy)), xx.dtype)

    loss = np.sum(yw)

    #print "min_pt {0} max_pt {1} coverage {2} thresh {3} loss {4}".format(self.min_pt, self.max_pt, self.coverage, thresh, loss)

    if isinstance(loss, list):
      return -loss[0]
    return -loss
