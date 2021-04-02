# graphDataGeneration.py

from graph_nets import utils_tf

from model.readRobot import *
from model.magnetoDefinition import *

#NodeIdDict = {} # (k,v) - (link name, node number) -> MagnetoGraphNode
#EdgeIDDict = {} # (k,v) - (joint name, edge number) -> MagnetoGraphEdge
#EdgeRecieverIDDict = {}
#EdgeSenderIDDict = {}


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
        #NodeIdDict[link.name ] = len(nodes) # need later for edge
        nodes.append(LinkData(link.inertial).data)
        
    
  for joint in robot.joints:
    for fn in FootNames:
      if( fn in joint.name and joint.joint_type != 'fixed'):
        #print(joint.name + ': ' + joint.parent + ' to ' + joint.child)
        #print(str(NodeIdDict[joint.parent]) + ',' + str(NodeIdDict[joint.child]) )
        #EdgeIDDict[joint.name ] = len(edges) # need later for edge
        #EdgeSenderIDDict[joint.name] = MagnetoGraphNode[joint.parent]
        #EdgeRecieverIDDict[joint.name] = MagnetoGraphNode[joint.child]
        edges.append(JointData(joint, ActuatedJoints).data)
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

def make_all_runnable_in_session(*args):
  """Apply make_runnable_in_session to an iterable of graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


# print(NodeIdDict)
# print(EdgeIDDict)
# for NodeName in NodeIdDict:
#     print(NodeName)
#     print(MagnetoLink[NodeName])
