from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


from matplotlib import pyplot as plt



try:
  import seaborn as sns
except ImportError:
  pass
else:
  sns.reset_orig()

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)