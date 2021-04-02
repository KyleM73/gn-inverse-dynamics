from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time
import math

import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models

from model.magnetoDefinition import *
from graphDataGeneration import *

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

###########################################################


def string_to_number(str):
  if("." in str):
    try:
      res = float(str)
    except:
      res = str  
  elif("-" in str):
    res = int(str)
  elif(str.isdigit()):
    res = int(str)
  else:
    res = str
  return(res)

def string_to_list(str):
  return [string_to_number(element) 
          for element in str.split()]


def read_trajectory():
  # q, q_des, dotq, dotq_des, trq, contact_al, f_mag_al, base_ori 
  f_q = open("magneto/data/q_sen.txt")
  f_dq = open("magneto/data/qdot_sen.txt")
  f_trq = open("magneto/data/trq.txt")

  f_base_pos = open("magneto/data/pose_base.txt")
  f_base_ori = open("magneto/data/rot_base.txt")

  f_mag_al = open("magneto/data/magnetic_AL_foot_link.txt")
  f_mag_ar = open("magneto/data/magnetic_AR_foot_link.txt")
  f_mag_bl = open("magneto/data/magnetic_BL_foot_link.txt")
  f_mag_br = open("magneto/data/magnetic_BR_foot_link.txt")

  q_des = string_to_list(f_q.readline())
  dq_des = string_to_list(f_dq.readline())  

  traj_dicts = [] 
  for q_line, dq_line, trq_line, bp_line, bo_line, fmal_line, fmar_line, fmbl_line, fmbr_line \
    in zip(f_q, f_dq, f_trq, f_base_pos, f_base_ori, f_mag_al, f_mag_ar, f_mag_bl, f_mag_br) : 
    
    q = q_des
    dq = dq_des
    q_des = string_to_list(q_line)  
    dq_des = string_to_list(dq_line)  
    trq = string_to_list(trq_line)
    #bp = string_to_list(bp_line)  
    bo = string_to_list(bo_line)  
    fmal = string_to_list(fmal_line)    
    fmar = string_to_list(fmar_line)  
    fmbl = string_to_list(fmbl_line)  
    fmbr = string_to_list(fmbr_line)

    f_contact_threshold = 50
    cal =   int( math.fabs( fmal[-1] ) > f_contact_threshold ) 
    car =   int( math.fabs( fmar[-1] ) > f_contact_threshold ) 
    cbl =   int( math.fabs( fmbl[-1] ) > f_contact_threshold ) 
    cbr =   int( math.fabs( fmbr[-1] ) > f_contact_threshold ) 

    traj_dicts.append( { 'q': q,
                    'q_des' : dq,
                    'dq' : q_des,
                    'dq_des' : dq_des,
                    'trq' : trq,
                    'contact_al' : cal,
                    'contact_ar' : car,
                    'contact_bl' : cbl,
                    'contact_br' : cbr,
                    'f_mag_al' : fmal,
                    'f_mag_ar' : fmar,
                    'f_mag_bl' : fmbl,
                    'f_mag_br' : fmbr,
                    'base_ori' : bo} )

  #print("size of traj dicts = " + str(len(traj_dicts)))
  #print(traj_dicts[-1])

  traj_dicts = traj_dicts[500:3000]
  return traj_dicts

#read_trajectory()


def traj_to_graph(traj_dict):
  '''
  Arg 
    traj_dict : dict( q, q_des, dq, dq_des, trq, 
                      contact_al/ar/bl/br, f_mag_al/ar/bl/br, base_ori )

  Returns 
    dynamic_graph : graph that has {global: base_ori,
                                    node : contact, magforce
                                    edge : q,dq,q_des,dq_des}
    target_edge_tr : list of target edge feature
  '''
  dyn_nodes = []
  dyn_edges = []
  ContactLink = {'AR_foot_link_3' : ['contact_ar', 'f_mag_ar'] ,
                 'BR_foot_link_3' : ['contact_br', 'f_mag_br'] , 
                 'AL_foot_link_3' : ['contact_al', 'f_mag_al'] , 
                 'BL_foot_link_3' : ['contact_bl', 'f_mag_bl'] }

  for linkname in MagnetoGraphNode:
    if(linkname in ContactLink):
      data = [ traj_dict[ContactLink[linkname][0]] ]
      data.extend(traj_dict[ContactLink[linkname][1]])
    else:
      data = [0, 0.0, 0.0, 0.0]

    dyn_nodes.append(data)

  target_edge_tr = []
  for jointname in MagnetoGraphEdge:
    data = [ traj_dict['q'][ MagnetoJoint[jointname] ] ]
    data.append(traj_dict['dq'][ MagnetoJoint[jointname] ])
    data.append(traj_dict['q_des'][ MagnetoJoint[jointname] ])
    data.append(traj_dict['dq_des'][ MagnetoJoint[jointname] ])
    dyn_edges.append(data)
    target_edge_tr.append([traj_dict['trq'][ MagnetoJoint[jointname] ]])

  dynamic_graph = magneto_graph(traj_dict['base_ori'], dyn_nodes, dyn_edges)
  
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
    if(len(nodes) is 0):
      nodes = graph['nodes'][:]
    else:
      nodes_temp = graph['nodes'][:]
      for node, node_temp in zip(nodes, nodes_temp):
        node.extend(node_temp)

    # concatenate edge
    if(len(edges) is 0):
      edges = graph['edges'][:]
    else:
      edges_temp = graph['edges'][:]
      for edge, edge_temp in zip(edges, edges_temp):
        edge.extend(edge_temp) 

  return magneto_graph(global_feature, nodes, edges)

##############################

def generate_graphs_dicts(rand, batch_size, traj_idx_min_max, static_graph, traj_dicts):
  
  input_graphs_dicts = []
  target_graphs_dicts = []

  emptynode = [[0]]*len(MagnetoGraphNode)

  traj_idx_start = rand.randint(*traj_idx_min_max)

  for traj_idx in range(traj_idx_start, traj_idx_start + batch_size):

    dynamic_graph, target_edge = traj_to_graph(traj_dicts[traj_idx])
    input_graph = concat_graph([dynamic_graph, static_graph])
    output_grpah = magneto_graph([], emptynode, target_edge)

    input_graphs_dicts.append(input_graph)
    target_graphs_dicts.append(output_grpah)

  return input_graphs_dicts, target_graphs_dicts

def create_placeholders(rand, batch_size, traj_idx_min_max_tr, 
                        static_graph, trajectory):
  input_graphs_dicts, target_graphs_dicts = generate_graphs_dicts(
                    rand, batch_size, traj_idx_min_max_tr, 
                    static_graph, trajectory)
  input_ph = utils_tf.placeholders_from_data_dicts(input_graphs_dicts)
  target_ph = utils_tf.placeholders_from_data_dicts(target_graphs_dicts)

  # print('input_graphs_dicts #########################')
  # print(input_graphs_dicts[0])
  # print(type(input_graphs_dicts[0]))
  # print('target_graphs_dicts #########################')
  # print(target_graphs_dicts[0])
  # print(type(target_graphs_dicts[0]))
  # print('input_ph #########################')
  # print(input_ph)
  # print(type(input_ph))
  # print('target_ph #########################')
  # print(target_ph)
  # print(type(target_ph))
  # print('#########################')


  return input_ph, target_ph


def create_feed_dict(rand, batch_size, traj_idx_min_max_tr,
                      static_graph, trajectory, 
                      input_ph, target_ph):
  input_graphs_dicts, target_graphs_dicts = generate_graphs_dicts(
                    rand, batch_size, traj_idx_min_max_tr, 
                    static_graph, trajectory)
  input_graphs = utils_np.data_dicts_to_graphs_tuple(input_graphs_dicts)
  target_graphs = utils_np.data_dicts_to_graphs_tuple(target_graphs_dicts)
  feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
  return feed_dict



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
