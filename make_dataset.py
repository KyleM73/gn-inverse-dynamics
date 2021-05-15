from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os 

import tensorflow as tf

from model.magnetoDefinition import *
from model.magnetoGraphGeneration import *
from functions import *
from utils.myutils import *

###########################################################
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
datasetpath = CURRENT_DIR_PATH + '/dataset'

batch_size_tr = 1024 #256
###########################################################

# set data
gen = get_trajectory_data()
trajDataSet_signature = get_traj_specs_from_graphs_tuples(next(gen))
trajDataSet = tf.data.Dataset.from_generator(get_trajectory_data,
                        output_signature = trajDataSet_signature )
dataset = trajDataSet.batch(batch_size_tr, drop_remainder=True)

###########################################################

log_with_time("save satrt")
dataset = tf.data.experimental.save(dataset, datasetpath)
log_with_time("save end")

