from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import itertools
import time

import numpy as np
import tensorflow as tf

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models

from graphDataGeneration import *
from functions_ph import *

from model.magnetoDefinition import *


SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

###########################################################

tf.reset_default_graph()

seed = 2
rand = np.random.RandomState(seed=seed)

# Model parameters.
num_processing_steps_tr = 5
num_processing_steps_te = 5

# Data / training parameters.
num_training_iterations = 5000
batch_size_tr = 800 #256
batch_size_ge = 100
num_time_steps = 50
step_size = 0.001

###########################################################
# LOAD DATA

# Read Trajectory Data and construct graphs for training
# traj_dicts : q, q_des, dotq, dotq_des, trq, foot_ct, base_ori 
traj_dicts = read_trajectory() 
N_traj = len(traj_dicts)
# print("traj_dicts size = ")
# print( len(traj_dicts))

# Base graphs for training. (static graph)
static_graph = magneto_base_graph('magneto/model/MagnetoSim_Dart.urdf') # data_dicts

traj_idx_min_max_tr = (0, N_traj-batch_size_tr)
print(traj_idx_min_max_tr)
###########################################################

# Data.
# Input and target placeholders.
input_ph, target_ph = create_placeholders(rand, batch_size_tr, 
                  traj_idx_min_max_tr, static_graph, traj_dicts)

# Connect the data to the model.
# Instantiate the model.
model = models.EncodeProcessDecode(edge_output_size=1)
# A list of outputs, one per processing step.
output_ops_tr = model(input_ph, num_processing_steps_tr)
# print(len(output_ops_tr))
# print(type(output_ops_tr))
# print('#########################')
# for ouput in output_ops_tr:
#     print(ouput)
#     print(type(ouput))
# print('#########################')

# Training Loss
loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
# Loss across processing steps.
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr

# Optimizer.
learning_rate = 1e-2 #1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

# Lets an iterable of TF graphs be output from a session as NP graphs.
input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

# Session.
try:
  sess.close()
except NameError:
  pass
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train
log_every_seconds = 10
start_time = time.time()
last_log_time = -10

losses_tr=[]

for iteration in range(0, num_training_iterations):
  feed_dict = create_feed_dict(rand, batch_size_tr, traj_idx_min_max_tr,
                      static_graph, traj_dicts, 
                      input_ph, target_ph)

  train_values = sess.run({
      "step": step_op,
      "target": target_ph,
      "loss": loss_op_tr,
      "outputs": output_ops_tr
  }, feed_dict=feed_dict)

  the_time = time.time()
  elapsed_since_last_log = the_time - last_log_time
  elapsed = time.time() - start_time
  print(" T {:.1f}, Ltr {:.4f}".format(elapsed, train_values["loss"]) )

  if elapsed_since_last_log > log_every_seconds:
    last_log_time = the_time
    losses_tr.append(train_values["loss"])
    print(train_values["target"].edges - train_values["outputs"][-1].edges)
    print(len(train_values["target"].edges))

# # Test
# randIdxList = []
# for i in range(5):
#   randIdxList.append(random.randint(2000,2020))

# for test_idx in randIdxList:
#   test_traj = traj_dicts[test_idx]
#   # build dynamic graph from trajectory data
#   dynamic_graph, target_edge_te = traj_to_graph(test_traj) # add noise later?

#   # graph concatentation
#   input_graph = concat_graph([dynamic_graph, static_graph])    
#   input_graph_te = utils_tf.data_dicts_to_graphs_tuple([input_graph])

#   # obtain predicted output
#   predicted_ops_te = model(input_graph_te, num_processing_steps_te) # 1d list
#   target_edge_te = tf.convert_to_tensor(target_edge_te, 
#                               dtype=predicted_ops_te[0].edges.dtype)
  
#   test_values = sess.run({
#       "target_edge_te": target_edge_te,
#       "predicted_ops_te": predicted_ops_te,
#   })

#   print(test_values["target_edge_te"])
#   print(test_values["predicted_ops_te"])


