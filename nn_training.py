import datetime
import sys

from nn_logging import getLogger
logger = getLogger()

from nn_models import save_my_model


# See https://stackoverflow.com/q/616645

class TrainingLog(object):
  def __init__(self):
    import os
    import sys
    import tempfile
    fd, name = tempfile.mkstemp(suffix='.txt', prefix='keras_output_', dir='.', text=True)
    self.file = os.fdopen(fd, 'w')
    self.name = name
    self.stdout = sys.stdout
  def __del__(self):
    self.file.close()
  def write(self, msg):
    self.file.write(msg)
  def flush(self):
    self.file.flush()


def train_model(model, x, y, model_name='model', batch_size=None, epochs=1, verbose=1, callbacks=None,
                validation_split=0., shuffle=True, class_weight=None, sample_weight=None):

  start_time = datetime.datetime.now()
  logger.info('Begin training ...')

  # Redirect sys.stdout
  log = TrainingLog()
  sys.stdout = log

  history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                      validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight)
  save_my_model(model, name=model_name)

  # Restore sys.stdout
  sys.stdout = log.stdout

  logger.info('Done training. Time elapsed: {0} sec'.format(str(datetime.datetime.now() - start_time)))
  return history


def train_model_sequential(model, x, y, model_name='model', batch_size=None, epochs=1, verbose=1, callbacks=None,
                           validation_split=0., shuffle=True, class_weight=None, sample_weight=None):
  start_time = datetime.datetime.now()
  logger.info('Begin training ...')

  # Redirect sys.stdout
  log = TrainingLog()
  sys.stdout = log

  history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                      validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight)
  save_my_model(model, name=model_name)

  # Restore sys.stdout
  sys.stdout = log.stdout

  logger.info('Done training. Time elapsed: {0} sec'.format(str(datetime.datetime.now() - start_time)))
  return history

