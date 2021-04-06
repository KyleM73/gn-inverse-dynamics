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
from graph_nets.demos_tf2 import models

from model.magnetoDefinition import *
from graphDataGeneration import *
from functions import *

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
seed = 2
rand = np.random.RandomState(seed=seed)

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

# LOAD DATA
traj_dicts = read_trajectory() 
traj_dicts = traj_dicts[500:3000]
N_traj = len(traj_dicts)
static_graph = magneto_base_graph('magneto-tf2/model/MagnetoSim_Dart.urdf') # data_dicts
traj_idx_min_max_tr = (0, N_traj-batch_size_tr)


# Data.
@tf.function
def get_data():
  # graph_tuple data
  inputs_tr, targets_tr, = create_graph_tuples(rand, batch_size_tr, 
                  traj_idx_min_max_tr, static_graph, traj_dicts)

  return inputs_tr, targets_tr


#####################################################################################

model = models.EncodeProcessDecode(edge_output_size=1)
inputs_tr, targets_tr = get_data()
output_tr = model(inputs_tr, num_processing_steps_tr)

print("--------------------------1-------------------")
print(type(model.variables))

for tfvar_model in model.variables:
  if "EncodeProcessDecode/MLPGraphNetwork/graph_network/edge_block/mlp/linear_0/b" in tfvar_model.name:
    print(type(tfvar_model))
    print(tfvar_model)

#####################################################################################

loaded = tf.saved_model.load( CURRENT_DIR_PATH + "/saved_model")

print("--------------------------2-------------------")

for tfvar_load in loaded.all_variables:
  print("loaded model variable name : " + tfvar_load.name)
  for tfvar_model in model.variables:
    if tfvar_model.name == tfvar_load.name:
      tfvar_model.assign(tfvar_load.value())
      print(tfvar_load.name + " is changed")


print("--------------------------3-------------------")
for tfvar_model in model.variables:
  if "EncodeProcessDecode/MLPGraphNetwork/graph_network/edge_block/mlp/linear_0/b" in tfvar_model.name:
    print(type(tfvar_model))
    print(tfvar_model)



#####################################################################################

# Optimizer.
learning_rate = 1e-3
optimizer = snt.optimizers.Adam(learning_rate)

def update_step(inputs_tr, targets_tr):
  with tf.GradientTape() as tape:
    output_ops_tr = model(inputs_tr, num_processing_steps_tr)
    # Loss.
    loss_ops_tr = create_loss_ops(targets_tr, output_ops_tr)
    # loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
    loss_tr = tf.math.reduce_sum(loss_ops_tr) / num_processing_steps_tr   

  gradients = tape.gradient(loss_tr, model.trainable_variables)
  optimizer.apply(gradients, model.trainable_variables)
  return output_ops_tr, loss_tr


# Get some example data that resembles the tensors that will be fed
# into update_step():
example_input_data, example_target_data = get_data()

# Get the input signature for that function by obtaining the specs
input_signature_tr = [
  utils_tf.specs_from_graphs_tuple(example_input_data),
  utils_tf.specs_from_graphs_tuple(example_target_data)
]

# Compile the update function using the input signature for speedy code.
compiled_update_step = tf.function(update_step, input_signature=input_signature_tr)

# Train
log_every_seconds = 20
TOTAL_TIMER = Timer()
LOG_TIMER = Timer()

losses_tr=[]

@tf.function(input_signature=utils_tf.specs_from_graphs_tuple(example_input_data))
def inference(x):
  return model(x, num_processing_steps_tr)

min_loss = [2e-2]

for iteration in range(0, num_training_iterations):
  inputs_tr, targets_tr = get_data()
  outputs_tr, loss_tr = compiled_update_step(inputs_tr, targets_tr)

  if LOG_TIMER.check(log_every_seconds):
    elapsed = TOTAL_TIMER.elapsed()
    losses_tr.append(loss_tr)
    print(" T {:.1f}, Ltr {:.4f}".format(elapsed, loss_tr) )
    # print(targets_tr.edges - outputs_tr[-1].edges)

    # print(model.variables)

    if(min_loss[-1] >  loss_tr):
      min_loss.append(loss_tr)    
      to_save = snt.Module()
      # to_save.inference = inference #inference
      to_save.all_variables = list(model.variables)    
      tf.saved_model.save(to_save, CURRENT_DIR_PATH + "/saved_model")


      print(" model saved " )