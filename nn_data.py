import numpy as np

#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nn_logging import getLogger
logger = getLogger()

from nn_encode import Encoder


# ______________________________________________________________________________
def muon_data(filename, adjust_scale=0, reg_pt_scale=1.0):
  try:
    logger.info('Loading muon data ...')
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
def pileup_data(filename, adjust_scale=0, reg_pt_scale=1.0, discr_pt_cut=0.):
  try:
    logger.info('Loading pileup data ...')
    loaded = np.load(filename)
    the_variables = loaded['variables']
    the_parameters = np.zeros((the_variables.shape[0], 3), dtype=np.float32)
    the_auxiliaries = loaded['aux']
    logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
    logger.info('Loaded the auxiliary info with shape {0}'.format(the_auxiliaries.shape))
  except:
    logger.error('Failed to load data from file: {0}'.format(filename))

  assert(the_auxiliaries.shape[1] == 4)  # jobid, ievt, highest_part_pt, highest_track_pt

  # Veto events with a muon with pT > 14 GeV
  sel = the_auxiliaries[:,2] > discr_pt_cut
  the_variables = the_variables[~sel]
  the_parameters = the_parameters[~sel]
  the_auxiliaries = the_auxiliaries[~sel]

  encoder = Encoder(the_variables, the_parameters, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale)
  x, y, w, x_mask = encoder.get_x(), encoder.get_y(), encoder.get_w(), encoder.get_x_mask()
  assert(np.isfinite(x).all())
  aux = the_auxiliaries
  return x, aux, w, x_mask


def pileup_data_split(filename, adjust_scale=0, reg_pt_scale=1.0, discr_pt_cut=0., test_job=50):
  x, aux, w, x_mask = pileup_data(filename, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, discr_pt_cut=discr_pt_cut)

  # Split dataset in training and testing
  split = aux[:,0] < test_job
  x_train, x_test, aux_train, aux_test, w_train, w_test, x_mask_train, x_mask_test = x[split], x[~split], aux[split], aux[~split], w[split], w[~split], x_mask[split], x_mask[~split]
  logger.info('Loaded # of training and testing events: {0}'.format((x_train.shape[0], x_test.shape[0])))
  return x_train, x_test, aux_train, aux_test, w_train, w_test, x_mask_train, x_mask_test
