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
# import gn_models as models
import gn_models as models


from model.magnetoDefinition import *
from model.magnetoGraphGeneration import *
from functions import *

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
#####################################################################################

# Model parameters.
num_processing_steps_tr = 3
num_processing_steps_ge = 3

# Data / training parameters.
num_training_iterations = 5000000
batch_size_tr = 200 #256
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
learning_rate = 1e-3
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
print("============ def functions =============")
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

# Compile the update function using the input signature for speedy code.
compiled_update_step = tf.function(update_step, input_signature=traj_signature)

log_f = open(CURRENT_DIR_PATH + "/results/time_per_loss_load.csv", 'a')

def save_data(time, loss, log_f):
  # print("save data")
  # log_f.write(str(time))
  # log_f.write(', ')
  # log_f.write(str(loss.numpy()))  
  log_f.write(" {:.1f}, {:.4f} \n".format(elapsed, batch_loss_sum) )

###################################################
# Train
log_every_seconds = 20
TOTAL_TIMER = Timer()
LOG_TIMER = Timer()

losses_tr=[]

epoch = 0
batch_iter = 0
min_loss = [150] #0.02
print("============ start training =============")

for iteration in range(0, num_training_iterations):
  # 1 epoch
  batch_iter = 0
  batch_loss_sum = 0
  for train_traj_tr in dataset :
    inputs_tr = graph_reshape(train_traj_tr[0])
    targets_tr = graph_reshape(train_traj_tr[1])
    outputs_tr, loss_tr = compiled_update_step(inputs_tr, targets_tr)
    # outputs_tr, loss_tr = update_step(inputs_tr, targets_tr)
    batch_loss_sum = batch_loss_sum + loss_tr
    # print("batch_iter = {:02d}".format(batch_iter))
    # batch_iter = batch_iter+1

  print("epoch_iter = {:02d}".format(epoch))
  epoch = epoch+1


  
  if LOG_TIMER.check(log_every_seconds):
    elapsed = TOTAL_TIMER.elapsed()
    losses_tr.append(batch_loss_sum)
    print(" T {:.1f}, Ltr {:.4f}".format(elapsed, batch_loss_sum) )
    # print(targets_tr.edges - outputs_tr[-1].edges)

    # print(model.variables)

    if(min_loss[-1] >  batch_loss_sum):
      min_loss.append(batch_loss_sum)
      to_save = snt.Module()
      # to_save.inference = inference #inference
      to_save.all_variables = list(model.variables)    
      tf.saved_model.save(to_save, CURRENT_DIR_PATH + "/saved_model")
      
      save_data(elapsed, batch_loss_sum, log_f)
      print(" model saved " )


      print("ouput = ")
      print(outputs_tr[-1].edges)
      print("target = ")
      print(targets_tr.edges)
      print("diff = ")
      print(outputs_tr[-1].edges- targets_tr.edges)
      print(" ================================================ " )