"""
export SCRAM_ARCH=slc6_amd64_gcc630
cmsrel CMSSW_9_3_6
cd CMSSW_9_3_6/src
cmsenv
cd ../..
#
virtualenv venv
# edit venv/bin/activate to set PYTHONPATH
source venv/bin/activate
pip install scikit-optimize

# More info about sklearn/skopt
#   http://scikit-learn.org/stable/modules/cross_validation.html
#   http://scikit-learn.org/stable/modules/grid_search.html
#   https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/sklearn-gridsearchcv-replacement.ipynb
"""

import sklearn
import skopt


# ______________________________________________________________________________
from nn_globals import *

from nn_encode import nlayers, nvariables

from nn_data import muon_data, pileup_data, muon_data_split, pileup_data_split

from nn_models import create_model, create_model_sequential, \
                      lr_decay, modelbestcheck, modelbestcheck_weights

from nn_training import train_model, train_model_sequential


use_hpe = ('SLURM_JOB_ID' in os.environ)
if use_hpe:
  infile_muon = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tba.14.npz'
  infile_pileup = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tbd.14.npz'


# ______________________________________________________________________________
# Import muon data
x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test = \
    muon_data_split(infile_muon, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, test_size=0.275)

# ______________________________________________________________________________
# Create KerasRegressor

from nn_models import NewKerasRegressor
estimator = NewKerasRegressor(build_fn=create_model_sequential, reg_pt_scale=reg_pt_scale, min_pt=20., max_pt=22., coverage=90.,
                              nvariables=nvariables, lr=learning_rate,
                              epochs=5, batch_size=256, verbose=0)
callbacks_list = [lr_decay,modelbestcheck]

# ______________________________________________________________________________
# Create BayesSearchCV

#FIXME

from sklearn.model_selection import cross_val_score, KFold
from skopt.space import Space, Real, Integer, Categorical
from skopt import BayesSearchCV
search_spaces = {
  #'lr': Real(1e-4, 1e-2, prior='log-uniform'),  # 'lr': Real(1e-4, 2e-3),
  'batch_size': Categorical([128, 256, 512, 1024, 2048, 4096, 8192]),
  #'nodes1': Integer(4, 256),
  #'nodes2': Integer(4, 256),
  #'nodes3': Integer(4, 256),
}

#opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, scoring='neg_mean_squared_error', fit_params=dict(callbacks=callbacks_list))
opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, scoring=None, fit_params=dict(callbacks=callbacks_list))
#opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, n_points=1, cv=KFold(5), n_jobs=5, scoring=None, return_train_score=True)

# ______________________________________________________________________________
# Fit

debug = True

if debug:
  estimator.model = estimator.build_fn(**estimator.filter_sk_params(estimator.build_fn))

  from keras.models import Sequential
  fit_args = estimator.filter_sk_params(Sequential.fit)
  print fit_args

  history = estimator.model.fit(x_train, y_train, **fit_args)
  print history.history

  print estimator.score(x_test, y_test)
  print

else:
  opt.fit(x_train, y_train)


# ______________________________________________________________________________
# Results

print("Best: %f using %s" % (opt.best_score_, opt.best_params_))
opt.best_estimator_.model.summary()
means = opt.cv_results_['mean_test_score']
stds = opt.cv_results_['std_test_score']
params = opt.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param))
print

print(opt)
print(opt.cv_results_)
print

print("Test: %f using %s" % (opt.score(x_test, y_test), opt.best_params_))
print

logger.info('DONE')
