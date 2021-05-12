from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time
import os 


import numpy as np
import tensorflow as tf
import sonnet as snt 

print(tf.__version__)
print(snt.__version__)

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
import gn_models as models

from model.magnetoDefinition import *
from model.magnetoGraphGeneration import *
from functions import *

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
#####################################################################################

# Model parameters.
num_processing_steps_tr = 5
num_processing_steps_ge = 5

# Data / training parameters.
num_training_iterations = 5000000
batch_size_tr = 500 #256
batch_size_ge = 100
num_time_steps = 50
step_size = 0.001

###########################################################
# LOAD DATA
print("============ LOAD DATA =============")
# Base graphs for training. (static graph - link/joint info)
static_graph = magneto_base_graph(CURRENT_DIR_PATH + '/model/MagnetoSim_Dart.urdf') # data_dicts

# construct trajectory dataset
gen = get_trajectory_data()
trajDataSet_signature = get_traj_specs_from_graphs_tuples(next(gen))
trajDataSet = tf.data.Dataset.from_generator(get_trajectory_data,
                        output_signature = trajDataSet_signature )

dataset = trajDataSet.batch(batch_size_tr, drop_remainder=True)
for train_traj_tr in dataset :
    inputs_tr = graph_reshape(train_traj_tr[0])
    targets_tr = graph_reshape(train_traj_tr[1])
    break

traj_signature = (
  utils_tf.specs_from_graphs_tuple(inputs_tr),
  utils_tf.specs_from_graphs_tuple(targets_tr)
)
#####################################################################################
# SET MODEL
print("============ set models =============")

# Optimizer.
learning_rate = 1e-2
optimizer = snt.optimizers.Adam(learning_rate)

model = models.EncodeProcessDecode(edge_output_size=1)

output_ops_tr = model(inputs_tr, num_processing_steps_tr)

print("1. initial model -------------------")
print(type(model.variables))

for tfvar_model in model.variables:
  if "EncodeProcessDecode/MLPGraphNetwork/graph_network/edge_block/mlp/linear_0/b" in tfvar_model.name:
    print(type(tfvar_model))
    print(tfvar_model)

print("2. load model and assign value -------------------")
loaded = tf.saved_model.load( CURRENT_DIR_PATH + "/saved_model")
for tfvar_load in loaded.all_variables:
  # print("loaded model variable name : " + tfvar_load.name)
  for tfvar_model in model.variables:
    if tfvar_model.name == tfvar_load.name:
      tfvar_model.assign(tfvar_load.value())
      # print(tfvar_load.name + " is changed")

print("3. loadded model -------------------")
for tfvar_model in model.variables:
  if "EncodeProcessDecode/MLPGraphNetwork/graph_network/edge_block/mlp/linear_0/b" in tfvar_model.name:
    print(type(tfvar_model))
    print(tfvar_model)



#####################################################################################

# DIFF FUCTION

@tf.function(input_signature=traj_signature)
def diff_model(inputs_tr, targets_tr):
  output_ops_tr = model(inputs_tr, num_processing_steps_tr)
  diff_ops_tr = [ targets_tr.edges - output_op_tr.edges
                  for output_op_tr in output_ops_tr ]
  return diff_ops_tr


# Check model accuracy

inputs_tr, targets_tr = get_data()
diff_ops_tr = diff_model(inputs_tr, targets_tr)

for diff_op_tr in diff_ops_tr:
  print("-----------------------")
  print(diff_op_tr)