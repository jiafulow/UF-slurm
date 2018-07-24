from nn_logging import getLogger
logger = getLogger()

# ______________________________________________________________________________
# Globals

adjust_scale = 3

learning_rate = 0.00113

mask_value = 100.

discr_pt_cut = 14.

reg_pt_scale = 100.

discr_loss_weight = 10.

add_noise = True

infile_muon = '../test7/histos_tba.14.npz'

infile_pileup = '../test7/histos_tbd.14.npz'

# ______________________________________________________________________________
# Import all the libs
import os
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
OLD_STDOUT = sys.stdout

logger.info('Using cmssw {0}'.format(os.environ['CMSSW_VERSION']))

import numpy as np
np.random.seed(2023)
logger.info('Using numpy {0}'.format(np.__version__))

import tensorflow as tf
logger.info('Using tensorflow {0}'.format(tf.__version__))

import keras
from keras import backend as K
#K.set_epsilon(1e-08)
#K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True)))
logger.info('Using keras {0}'.format(keras.__version__))
logger.info('.. list devices: {0}'.format(K.get_session().list_devices()))

import sklearn
logger.info('Using sklearn {0}'.format(sklearn.__version__))

import matplotlib.pyplot as plt
from matplotlib import colors
#%matplotlib inline

