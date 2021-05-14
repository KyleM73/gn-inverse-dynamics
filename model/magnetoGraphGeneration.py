# model.magnetoGraphGeneration.py

from graph_nets import utils_tf

import os
import tensorflow as tf
import math


import utils.myutils as myutils
from graph_nets import utils_tf

from model.readRobot import *
from model.magnetoDefinition import *
from utils.myutils import *


CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# function need for rotation invarianct descriptions
def shuffle_list(list_in):
    list_out = []
    if(len(list_in)>0):
        list_out = [list_in[-1]]
        list_out.extend(list_in[0:len(list_in)-1])
    return list_out

def shuffle_list_reverse(list_in):
    list_out = []
    if(len(list_in)>0):        
        list_out.extend(list_in[1:len(list_in)])
        list_out.append(list_in[0])
    return list_out

# graph generation
def magneto_base_graph(fn):
  """Define a basic magneto graph structure.
  Args:
    ct: contact 
    base_ori
    q_act, q_des
    dq_act, dq_des    

  Returns:
    data_dict: dictionary with globals, nodes, edges, receivers and senders
                to represent a structure like the one above.
  """
  robot = loadRobotURDF(fn)
  nodes = []
  edges = []
  receivers = []
  senders = []

  # nodes (v_k : Link Information)
  # mass, inertia, pos, ori
  FootNames = ['base_link','AR','BR','BL','AL']
  ActuatedJoints = ['coxa', 'femur', 'tibia']
  
  for link in robot.links:
    for fn in FootNames:
      if(fn in link.name and link.inertial.mass > 1e-3 ):
        nodes.append(LinkData(link.inertial).data)
        
  joints = []
  srcsj = [] 
  destj = []

  for joint in robot.joints:
    for fn in FootNames:
      if( fn in joint.name and joint.joint_type != 'fixed'):
        joints.append(joint)            
        srcsj.append(MagnetoGraphNode[joint.parent])
        destj.append(MagnetoGraphNode[joint.child])

  
  for link_idx in range(len(nodes)):
      # i : list of joint where the parent link == link_idx
      idxs = [i for i, x in enumerate(srcsj) if x==link_idx]

      if(len(idxs) > 1):
          idxs1 = shuffle_list(idxs)
          idxs2 = shuffle_list_reverse(idxs)
          for j, j1, j2 in zip( idxs, idxs1, idxs2 ):
              joint = joints[j]
              joint1 = joints[j1]
              joint2 = joints[j2]
              edges.append(EdgeData(joint, joint1, joint2, ActuatedJoints).data)
              senders.append(MagnetoGraphNode[joint.parent])
              receivers.append(MagnetoGraphNode[joint.child])
      
      elif(len(idxs) == 1):
          idxs1 = [i for i, x in enumerate(destj) if x==link_idx]
          for j, j1 in zip( idxs, idxs1 ):
              joint = joints[j]
              joint1 = joints[j1]
              edges.append(EdgeData(joint, joint1, joint1, ActuatedJoints).data)
              senders.append(MagnetoGraphNode[joint.parent])
              receivers.append(MagnetoGraphNode[joint.child])
        
        
  # edges (e_k : Joint Information)
  # axis, pos, ori, actuated

  return {
      "globals": [0., 0., -9.8], #gravity vector
      "nodes": nodes, #Linkdata
      "edges": edges, #Jointdata
      "receivers": receivers, # node number
      "senders": senders
  }



def magneto_graph(global_feature, nodes, edges):

  '''
    Assume edge feature list is following the magneto_static_graph order
  '''
  receivers = []
  senders = []
  
  for sender in MagnetoGraphEdgeSender:
    senders.append(MagnetoGraphEdgeSender[sender]) 

  for receiver in MagnetoGraphEdgeReceiver:
    receivers.append(MagnetoGraphEdgeReceiver[receiver]) 

  return {
      "globals": global_feature,
      "nodes": nodes,
      "edges": edges,
      "receivers": receivers,
      "senders": senders
  }



###########################################################
############        traj graph generation   ###############
###########################################################

def string_to_number(str):
  # if("." in str):
  #   try:
  #     res = float(str)
  #   except:
  #     res = str  
  # elif("-" in str):
  #   res = int(str)
  # elif(str.isdigit()):
  #   res = int(str)
  # else:
  #   res = str
  res = float(str)
  return(res)

def string_to_list(str):
  return [string_to_number(element) 
          for element in str.split()]

def get_trajectory_files(path, folder):
  # q, q_des, dotq, dotq_des, trq, contact_al, f_mag_al, base_ori
  f_q = open(path + folder + "/q_sen.txt")
  f_dq = open(path + folder + "/qdot_sen.txt")

  f_q_d = open(path + folder + "/q_des.txt")
  f_dq_d = open(path + folder + "/qdot_des.txt")
  
  f_trq = open(path + folder + "/trq.txt")

  f_base_ori = open(path + folder + "/rot_base.txt")

  f_mag_al = open(path + folder + "/magnetic_AL_foot_link.txt")
  f_mag_ar = open(path + folder + "/magnetic_AR_foot_link.txt")
  f_mag_bl = open(path + folder + "/magnetic_BL_foot_link.txt")
  f_mag_br = open(path + folder + "/magnetic_BR_foot_link.txt")
  return zip(f_q, f_q_d, f_dq, f_dq_d, f_trq, f_base_ori, f_mag_al, f_mag_ar, f_mag_bl, f_mag_br)

def traj_data_to_graph(q_line, qd_line, dq_line, dqd_line, trq_line,  
                      bo_line, fmal_line, fmar_line, fmbl_line, fmbr_line):

  q = string_to_list(q_line)
  dq = string_to_list(dq_line)
  q_des = string_to_list(qd_line)  
  dq_des = string_to_list(dqd_line)  
  trq = string_to_list(trq_line)

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

  traj_dict = { 'q': q,
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
                'base_ori' : bo }

  dynamic_graph, target_graph = traj_to_graph( traj_dict )

  dynamic_graph_tsr = utils_tf.data_dicts_to_graphs_tuple( [dynamic_graph] )
  target_graph_tsr = utils_tf.data_dicts_to_graphs_tuple( [target_graph] )
  dynamic_graph_tsr = dynamic_graph_tsr.replace(globals = tf.cast(dynamic_graph_tsr.globals, tf.float32))

  return (dynamic_graph_tsr, target_graph_tsr)

def get_trajectory_data_test():
  data_path = CURRENT_DIR_PATH + '/../dataMerged'
  test_folder = '/datafinal_test'
  files_zip = get_trajectory_files(path=data_path, folder=test_folder)

  for q_line, qd_line, dq_line, dqd_line, trq_line,\
      bo_line, fmal_line, fmar_line, fmbl_line, fmbr_line in files_zip:    
    
    yield traj_data_to_graph(q_line, qd_line, dq_line, dqd_line, trq_line,\
                      bo_line, fmal_line, fmar_line, fmbl_line, fmbr_line)

def get_trajectory_data():
  # print("get_trajectory_data")
  #f_q, f_q_d, f_dq, f_dq_d, f_trq, f_base_ori, f_mag_al, f_mag_ar, f_mag_bl, f_mag_br)
  lop_with_time("get_trajectory_data")
  data_path = CURRENT_DIR_PATH + '/../dataMerged'
  train_folder = '/datafinal'
  files_zip = get_trajectory_files(path=data_path, folder=train_folder)

  for q_line, qd_line, dq_line, dqd_line, trq_line,\
      bo_line, fmal_line, fmar_line, fmbl_line, fmbr_line in files_zip:    
    
    yield traj_data_to_graph(q_line, qd_line, dq_line, dqd_line, trq_line,
                      bo_line, fmal_line, fmar_line, fmbl_line, fmbr_line)


def get_traj_specs_from_graphs_tuples(graph_tuples):
  return (utils_tf.specs_from_graphs_tuple(graph_tuples[0]), 
            utils_tf.specs_from_graphs_tuple(graph_tuples[1]))
  
def get_traj_type_shape_from_dict(example_dict):
  traj_type={}
  traj_shape={}
  for key in example_dict:
    traj = example_dict[key]
    traj_type[key] = traj.dtype
    traj_shape[key] = traj.shape
  # print(example_dict)
  # print(traj_type)
  # print(traj_shape)
  return traj_type, traj_shape

def get_traj_type_shape_from_dicts(example_dicts):
  traj_types = []
  traj_shapes = []
  for example_dict in example_dicts:
    print(example_dict)
    traj_type, traj_shape = get_traj_type_shape_from_dict(example_dict)
    traj_types.append(traj_type)
    traj_shapes.append(traj_shapes)

  return traj_types, traj_shapes

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
    target_edge_tr.append([ float( traj_dict['trq'][ MagnetoJoint[jointname] ] )] )

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
  target_graph = magneto_graph([], [], target_edge_tr)
 
  return dynamic_graph, target_graph