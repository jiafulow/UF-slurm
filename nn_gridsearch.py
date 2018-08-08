"""
export SCRAM_ARCH=slc6_amd64_gcc630
cmsrel CMSSW_10_1_7
cd CMSSW_10_1_7/src
cmsenv
virtualenv venv
source venv/bin/activate
pip install -U pip
pip install scikit-optimize
cd ../..

# More info about sklearn/skopt
#   http://scikit-learn.org/stable/modules/cross_validation.html
#   http://scikit-learn.org/stable/modules/grid_search.html
#   https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/sklearn-gridsearchcv-replacement.ipynb
"""


# ______________________________________________________________________________
from nn_globals import *

from nn_encode import nlayers, nvariables

from nn_data import muon_data, pileup_data, muon_data_split, pileup_data_split, \
                    mix_training_inputs

from nn_models import create_model, create_model_bn, \
                      create_model_sequential, create_model_sequential_regularized, \
                      lr_decay, modelbestcheck, modelbestcheck_weights

from nn_training import train_model, train_model_sequential


# ______________________________________________________________________________
import skopt
logger.info('Using skopt {0}'.format(skopt.__version__))

use_hpe = ('SLURM_JOB_ID' in os.environ)

if use_hpe:
  infile_muon = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tba.16.npz'
  infile_pileup = '/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/histos_tbd.16.npz'


# ______________________________________________________________________________
# Import muon data
x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test = \
    muon_data_split(infile_muon, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, test_size=0.3)

# ______________________________________________________________________________
# Create KerasRegressor

from nn_models import NewKerasRegressor
estimator = NewKerasRegressor(build_fn=create_model_sequential, reg_pt_scale=reg_pt_scale,
                              min_pt=20., max_pt=22., coverage=90.,
                              nvariables=nvariables, lr=learning_rate, l1_reg=l1_reg, l2_reg=l2_reg,
                              epochs=20, batch_size=8192, verbose=0)
callbacks_list = [lr_decay,modelbestcheck]


# ______________________________________________________________________________
# Create GridSearchCV

from sklearn.model_selection import cross_val_score, cross_validate, KFold, GridSearchCV

nodes1 = [32,40,64,80,128]
nodes2 = [16,24,32,48]
nodes3 = [16,24,32,48]
lr = [0.001, 0.005, 0.01]
param_grid = dict(nodes1=nodes1, nodes2=nodes2, nodes3=nodes3, lr=lr)
print("param_grid: %r" % param_grid)

#opt = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_mean_squared_error',
#                   n_jobs=1, cv=None, verbose=0, return_train_score=False)
opt = GridSearchCV(estimator=estimator, param_grid=param_grid,
                   n_jobs=1, cv=None, verbose=0, return_train_score=False)
opt.fit(x_train, y_train)


# ______________________________________________________________________________
# Create BayesSearchCV

#from sklearn.model_selection import cross_val_score, cross_validate, KFold
#from skopt.space import Space, Real, Integer, Categorical
#from skopt import BayesSearchCV
#search_spaces = {
#  #'lr': Real(1e-4, 1e-2, prior='log-uniform'),  # 'lr': Real(1e-4, 2e-3),
#  'batch_size': Categorical([128, 256, 512, 1024, 2048, 4096, 8192]),
#  #'nodes1': Integer(4, 256),
#  #'nodes2': Integer(4, 256),
#  #'nodes3': Integer(4, 256),
#}

##opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, scoring='neg_mean_squared_error', fit_params=dict(callbacks=callbacks_list))
#opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, scoring=None, fit_params=dict(callbacks=callbacks_list))
##opt = BayesSearchCV(estimator=estimator, search_spaces=search_spaces, n_iter=5, n_points=1, cv=KFold(5), n_jobs=5, scoring=None, return_train_score=True)
#opt.fit(x_train, y_train)


# ______________________________________________________________________________
# Results

print("Best: %f using %s" % (opt.best_score_, opt.best_params_))
opt.best_estimator_.model.summary()
means = opt.cv_results_['mean_test_score']
stds = opt.cv_results_['std_test_score']
params = opt.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (+/-%f) with: %r" % (mean, stdev, param))
print

print(opt)
print(opt.cv_results_)
print

#print("Test: %f using %s" % (opt.score(x_test, y_test), opt.best_params_))
#print

logger.info('DONE')
