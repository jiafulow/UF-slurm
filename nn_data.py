import numpy as np

#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nn_logging import getLogger
logger = getLogger()

from nn_encode import Encoder


# ______________________________________________________________________________
def muon_data(filename, adjust_scale=0, reg_pt_scale=1.0):
  try:
    logger.info('Loading muon data from {0} ...'.format(filename))
    loaded = np.load(filename)
    the_variables = loaded['variables']
    the_parameters = loaded['parameters']
    logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
    logger.info('Loaded the parameters with shape {0}'.format(the_parameters.shape))
  except:
    logger.error('Failed to load data from file: {0}'.format(filename))

  encoder = Encoder(the_variables, the_parameters, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale)
  x, y, w, x_mask = encoder.get_x(), encoder.get_y(), encoder.get_w(), encoder.get_x_mask()
  assert(np.isfinite(x).all())
  return x, y, w, x_mask


def muon_data_split(filename, adjust_scale=0, reg_pt_scale=1.0, test_size=0.5):
  x, y, w, x_mask = muon_data(filename, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale)

  # Split dataset in training and testing
  x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test = train_test_split(x, y, w, x_mask, test_size=test_size)
  logger.info('Loaded # of training and testing events: {0}'.format((x_train.shape[0], x_test.shape[0])))
  return x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test


# ______________________________________________________________________________
def pileup_data(filename, adjust_scale=0, reg_pt_scale=1.0):
  try:
    logger.info('Loading pileup data from {0} ...'.format(filename))
    loaded = np.load(filename)
    the_variables = loaded['variables']
    the_parameters = np.zeros((the_variables.shape[0], 3), dtype=np.float32)
    the_auxiliaries = loaded['aux']
    logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
    logger.info('Loaded the auxiliary PU info with shape {0}'.format(the_auxiliaries.shape))
  except:
    logger.error('Failed to load data from file: {0}'.format(filename))

  assert(the_auxiliaries.shape[1] == 4)  # jobid, ievt, highest_part_pt, highest_track_pt

  encoder = Encoder(the_variables, the_parameters, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale)
  x, y, w, x_mask = encoder.get_x(), encoder.get_y(), encoder.get_w(), encoder.get_x_mask()
  assert(np.isfinite(x).all())
  aux = the_auxiliaries
  return x, aux, w, x_mask


def pileup_data_split(filename, adjust_scale=0, reg_pt_scale=1.0, test_job=50):
  x, aux, w, x_mask = pileup_data(filename, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale)

  # Split dataset in training and testing
  split = aux[:,0] < test_job
  x_train, x_test, aux_train, aux_test, w_train, w_test, x_mask_train, x_mask_test = x[split], x[~split], aux[split], aux[~split], w[split], w[~split], x_mask[split], x_mask[~split]
  logger.info('Loaded # of training and testing events: {0}'.format((x_train.shape[0], x_test.shape[0])))
  return x_train, x_test, aux_train, aux_test, w_train, w_test, x_mask_train, x_mask_test


# ______________________________________________________________________________
def mix_training_inputs(x_train, y_train, pu_x_train, pu_y_train, pu_aux_train, discr_pt_cut=14.):

  # Apply veto on PU events with a muon with pT > 14 GeV
  pu_x_train_tmp = ~(pu_aux_train[:,2] > discr_pt_cut)
  pu_x_train = pu_x_train[pu_x_train_tmp]
  pu_y_train = [pu_y_train[0][pu_x_train_tmp], pu_y_train[1][pu_x_train_tmp]]

  # Put together x_train & pu_x_train, y_train & pu_y_train
  assert(len(pu_y_train) == 2)
  assert(pu_x_train.shape[0] == pu_y_train[0].shape[0])
  assert(pu_x_train.shape[0] == pu_y_train[1].shape[0])
  num_samples = pu_x_train.shape[0]
  index_array = np.arange(num_samples)
  index_array_ext = np.tile(index_array, 14)  # 14 is chosen to make sure pu_x_train_ext has more entries than x_train
  pu_x_train_ext = pu_x_train[index_array_ext]
  pu_y_train_ext = [pu_y_train[0][index_array_ext], pu_y_train[1][index_array_ext]]

  assert(len(y_train) == 2)
  assert(x_train.shape[0] == y_train[0].shape[0])
  assert(x_train.shape[0] == y_train[1].shape[0])
  if not (x_train.shape[0] < pu_x_train_ext.shape[0]):
    raise Exception('pu_x_train_ext is required to have more entries than x_train. failed check {0} < {1}'.format(x_train.shape[0], pu_x_train_ext.shape[0]))
  num_samples = x_train.shape[0]
  index_array = np.arange(num_samples)
  #np.random.shuffle(index_array)

  try:
    from keras.engine.training import _make_batches as make_batches
  except ImportError:
    from keras.engine.training_utils import make_batches

  sample_batch_size = 128
  batches = make_batches(num_samples, sample_batch_size)

  x_train_new = np.zeros((num_samples*2, x_train.shape[1]), dtype=np.float32)
  y_train_new = [np.zeros((num_samples*2,), dtype=np.float32), np.zeros((num_samples*2,), dtype=np.float32)]

  for batch_index, (batch_start, batch_end) in enumerate(batches):
    batch_ids = index_array[batch_start:batch_end]
    x_train_new[batch_start*2:batch_start*2 + (batch_end-batch_start)] = x_train[batch_ids]
    x_train_new[batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_x_train_ext[batch_ids]
    y_train_new[0][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[0][batch_ids]
    y_train_new[0][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[0][batch_ids]
    y_train_new[1][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[1][batch_ids]
    y_train_new[1][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[1][batch_ids]

  logger.info('Mixed muon data with pileup data. x_train_new has shape {0}, y_train_new has shape {1},{2}'.format(x_train_new.shape, y_train_new[0].shape, y_train_new[1].shape))
  return x_train_new, y_train_new
