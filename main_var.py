from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random
import time
import numpy as np
import sonnet as snt
import tensorflow as tf

from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets.demos import models

from graphDataGeneration import *
from functions import *

from model.magnetoDefinition import *

########################################

tf.reset_default_graph()

# Model parameters.
num_processing_steps_tr = 2
num_processing_steps_te = 2

# Data / training parameters.
num_training_iterations = 5000
batch_size_tr = 256
batch_size_ge = 100
num_time_steps = 50
step_size = 0.001

# Create the model.
model = models.EncodeProcessDecode(edge_output_size=1)

# Base graphs for training. (static graph)
static_graph = magneto_base_graph('magneto/model/MagnetoSim_Dart.urdf') # data_dicts
# print(NodeIdDict)
# print(EdgeIDDict)
# for NodeName in NodeIdDict:
#     print(NodeName)
#     print(MagnetoLink[NodeName])



# Read Trajectory Data and construct graphs for training
# TODO 1
# q, q_des, dotq, dotq_des, trq, foot_ct, base_ori 
traj_dicts = read_trajectory() 
print("traj_dicts size = ")
print( len(traj_dicts))
# TODO 2 loss function
N_DATA = len(traj_dicts)
loss_op_tr = 0

traj_iteration = 0

for traj in traj_dicts:
    traj_iteration += 1
    if(traj_iteration > 1000 and traj_iteration < 1500):
      pass

      print( "# iteration : {:d}".format( traj_iteration ) )

      # build dynamic graph from trajectory data
      dynamic_graph, target_edge_tr = traj_to_graph(traj) # add noise later?

      # graph concatentation
      input_graph = concat_graph([dynamic_graph, static_graph])    
      input_graph_tr = utils_tf.data_dicts_to_graphs_tuple([input_graph])

      # obtain predicted output
      output_ops_tr = model(input_graph_tr, num_processing_steps_tr) # 1d list

      # Training loss
      target_edge_tr = tf.convert_to_tensor(target_edge_tr, 
                                        dtype=output_ops_tr[0].edges.dtype)
      loss_ops_tr = create_loss_ops(target_edge_tr, output_ops_tr)
      
      # Training loss across processing steps.
      loss_op_tr += sum(loss_ops_tr) / num_processing_steps_tr

# Optimizer.
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

input_graph_tr = make_all_runnable_in_session(input_graph_tr)

# Session.
try:
  sess.close()
except NameError:
  pass
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
elapsed = time.time()
train_values = sess.run({
    "step": step_op,
    "loss": loss_op_tr,
    "input_graph": input_graph_tr,
    "target_edges": target_edge_tr,
    "outputs": output_ops_tr
})
elapsed = time.time() - elapsed
print(" T {:.1f}, Ltr {:.4f}".format(elapsed, train_values["loss"]) )

# Test
randIdxList = []
for i in range(5):
  randIdxList.append(random.randint(2000,2020))

for test_idx in randIdxList:
  test_traj = traj_dicts[test_idx]
  # build dynamic graph from trajectory data
  dynamic_graph, target_edge_te = traj_to_graph(test_traj) # add noise later?

  # graph concatentation
  input_graph = concat_graph([dynamic_graph, static_graph])    
  input_graph_te = utils_tf.data_dicts_to_graphs_tuple([input_graph])

  # obtain predicted output
  predicted_ops_te = model(input_graph_te, num_processing_steps_te) # 1d list
  target_edge_te = tf.convert_to_tensor(target_edge_te, 
                              dtype=predicted_ops_te[0].edges.dtype)
  
  test_values = sess.run({
      "target_edge_te": target_edge_te,
      "predicted_ops_te": predicted_ops_te,
  })



  print(test_values["target_edge_te"])
  print(test_values["predicted_ops_te"])
