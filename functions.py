from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

import os

import networkx as nx
import numpy as np
from scipy import spatial
# from scipy.spatial.transform import Rotation as R

import tensorflow as tf

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
import gn_models as models

from model.magnetoDefinition import *
from model.magnetoGraphGeneration import *
# from utils.myutils import * 
import utils.myutils as myutils

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


###########################################################



class Timer():
  def __init__(self):
    self._start_time = time.time() 

  def reset(self):
    self._start_time = time.time()

  def elapsed(self):
    self._elapsed_time = time.time() - self._start_time
    return self._elapsed_time

  def check(self, check_time):
    if(self.elapsed() > check_time):
      self.reset()
      return True
    else:
      return False



###########################################################




def traj_to_graph(traj_dict):
  '''
  Arg 
    traj_dict : dict( q, q_des, dq, dq_des, trq, 
                      contact_al/ar/bl/br, f_mag_al/ar/bl/br, base_ori )

  Returns 
    dynamic_graph : graph that has {global: base_ori --> z-dir and mode(theta,180),
                                    node : contact, magforce
                                    edge : q,dq,q_des,dq_des}
    target_edge_tr : list of target edge feature
  '''
  dyn_globals = []
  dyn_nodes = []
  dyn_edges = []
  ContactLink = {'AR_foot_link_3' : ['contact_ar', 'f_mag_ar'] ,
                 'BR_foot_link_3' : ['contact_br', 'f_mag_br'] , 
                 'AL_foot_link_3' : ['contact_al', 'f_mag_al'] , 
                 'BL_foot_link_3' : ['contact_bl', 'f_mag_bl'] }

  # dyn_nodes
  for linkname in MagnetoGraphNode:
    if(linkname in ContactLink):
      data = [ traj_dict[ContactLink[linkname][0]] ]
      data.extend(traj_dict[ContactLink[linkname][1]])
    else:
      data = [0, 0.0, 0.0, 0.0]
    dyn_nodes.append(data)

  # dyn_edges
  target_edge_tr = []
  for jointname in MagnetoGraphEdge:
    data = [ traj_dict['q'][ MagnetoJoint[jointname] ] ]
    data.append(traj_dict['dq'][ MagnetoJoint[jointname] ])
    data.append(traj_dict['q_des'][ MagnetoJoint[jointname] ])
    data.append(traj_dict['dq_des'][ MagnetoJoint[jointname] ])
    dyn_edges.append(data)
    target_edge_tr.append([traj_dict['trq'][ MagnetoJoint[jointname] ]])

  # dyn_globals
  base_quat = traj_dict['base_ori'] # w,x,y,z
  base_rz = myutils.quat_to_rot_axis(base_quat,'z')
  base_rx = myutils.quat_to_rot_axis(base_quat,'x')
  base_rx_zero = np.cross(base_rz, [0.,1.,0.])
  base_rx_zero = myutils.normalize(base_rx_zero)
  delta_theta_rx = myutils.angle_between_axes(base_rx_zero, base_rx)
  delta_theta_rx = myutils.mod_angle(delta_theta_rx, [-np.pi/2, np.pi/2])

  dyn_globals.extend(base_rz)
  dyn_globals.append(delta_theta_rx)

  # magneto graph
  dynamic_graph = magneto_graph(dyn_globals, dyn_nodes, dyn_edges)
  
  return dynamic_graph, target_edge_tr


def concat_graph(graph_dicts):
  '''
  Arg:
    graph_dicts : list of graph dictionary
  concatenate graphs with the same topology
  '''
  global_feature = []
  nodes = []
  edges = []
  for graph in graph_dicts :
    # concatenate global
    if(type(graph['globals']) is list):
      global_feature.extend(graph['globals'])
    else:
      global_feature.append(graph['globals'])

    # concatenate node
    if(len(nodes) == 0):
      nodes = graph['nodes'][:]
    else:
      nodes_temp = graph['nodes'][:]
      for node, node_temp in zip(nodes, nodes_temp):
        node.extend(node_temp)

    # concatenate edge
    if(len(edges) == 0):
      edges = graph['edges'][:]
    else:
      edges_temp = graph['edges'][:]
      for edge, edge_temp in zip(edges, edges_temp):
        edge.extend(edge_temp) 

  return magneto_graph(global_feature, nodes, edges)

##############################

def generate_graphs_dicts(batch_size, traj_idx_min_max, static_graph, trajDataSet):
  
  input_graphs_dicts = []
  target_graphs_dicts = []
  traj_dicts = trajDataSet.batch(batch_size)

  for traj_dict in traj_dicts:
    # dynamic_graph, target_edge = traj_to_graph(traj_dicts[traj_idx])
    dynamic_graph, target_edge = traj_to_graph(tf.gather(traj_dicts,traj_idx))
    input_graph = concat_graph([dynamic_graph, static_graph])
    output_grpah = magneto_graph([], [], target_edge)

    input_graphs_dicts.append(input_graph)
    target_graphs_dicts.append(output_grpah)

  return input_graphs_dicts, target_graphs_dicts

def create_graph_tuples(batch_size, 
                  traj_idx_min_max, static_graph, trajDataSet):
    input_graphs_dicts, target_graphs_dicts = generate_graphs_dicts(
                batch_size, traj_idx_min_max, static_graph, trajDataSet )
    
    inputs_tr = utils_tf.data_dicts_to_graphs_tuple(input_graphs_dicts)
    outputs_tr = utils_tf.data_dicts_to_graphs_tuple(target_graphs_dicts)
    return inputs_tr, outputs_tr

##############################

def create_loss_ops(target_op, output_ops):
  ''' Create supervised loss operations from targets and outputs.
  Args:
    target_op: The tensor of target torque (edge).
    output_ops: The list of output graphs from the model.

  Returns:
    A list of loss values (tf.Tensor), one per output op.
  '''


  # loss_ops = [
  #     tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
  #     for output_op in output_ops
  # ]

  loss_ops = [tf.reduce_mean(
              tf.reduce_sum((output_op.edges - target_op.edges)**2, axis=-1))
              for output_op in output_ops   ]
  return loss_ops






##############################

# NODES = "nodes"
# EDGES = "edges"
# RECEIVERS = "receivers"
# SENDERS = "senders"
# GLOBALS = "globals"
# N_NODE = "n_node"
# N_EDGE = "n_edge"
# ALL_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS, N_NODE, N_EDGE)

def tensor_to_list(ts, axis=0):
  shape_list = ts.shape.as_list()
  split_size = shape_list[axis]

  ts_list = tf.split(value=ts, axis=axis, num_or_size_splits=split_size)
  return [ tf.squeeze(ts_split, [axis], name=None) for ts_split in ts_list]



def graph_split(graph_tuples_batch):

  nodes = tensor_to_list(graph_tuples_batch.nodes)

  edges = tensor_to_list(graph_tuples_batch.edges)
  globals_ = tensor_to_list(graph_tuples_batch.globals)

  receivers = tensor_to_list(graph_tuples_batch.receivers)
  senders = tensor_to_list(graph_tuples_batch.senders)

  n_node = tensor_to_list(graph_tuples_batch.n_node)
  n_edge = tensor_to_list(graph_tuples_batch.n_edge)

  n_graph = len(n_edge)

  return [
   graph_tuples_batch.replace(nodes=nodes[i], edges=edges[i], globals=globals_[i],
   receivers=receivers[i], senders=senders[i], n_node=n_node[i], n_edge=n_edge[i])
   for i in range(n_graph) ] 

def graph_reshape(graph_tuples_batch):
  graph_lists = graph_split(graph_tuples_batch)
  concat_graph_tuples = utils_tf.concat(graph_lists, axis=0)
  return concat_graph_tuples




