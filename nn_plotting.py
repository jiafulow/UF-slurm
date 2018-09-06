import numpy as np

import contextlib

from scipy.stats import beta

def find_sumw2_errors(y, w):
  sumw2 = y * np.square(w)
  return np.sqrt(sumw2)

def find_efficiency_errors(total_array, passed_array, level=0.682689492137):
  """Copied from ROOT TEfficiency::ClopperPearson()
  """
  assert(total_array.ndim == 1)
  assert(passed_array.ndim == 1)

  alpha = (1.0 - level) / 2
  lower_array = np.zeros(total_array.shape[0], dtype=np.float32)
  upper_array = np.zeros(total_array.shape[0], dtype=np.float32)

  for i, (total, passed) in enumerate(zip(total_array, passed_array)):
    if total == 0.:
      eff = 0.
    else:
      eff = np.true_divide(passed, total)
    if passed == 0.:
      l = 0.0
    else:
      l = beta.ppf(alpha, passed, total-passed+1)
    if passed == total:
      u = 1.0
    else:
      u = beta.ppf(1 - alpha, passed+1, total-passed)
    lower_array[i] = eff - l
    upper_array[i] = u - eff
  return np.vstack((lower_array, upper_array))


# Answer from https://stackoverflow.com/a/2891805
@contextlib.contextmanager
def np_printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)
