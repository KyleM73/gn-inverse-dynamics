from urdfpy import URDF
from geometry import axis_angle_from_rotation
from scipy.spatial.transform import Rotation as R
from model.mathfunctions import *
# links, joints, actuated_joints



def check_actuated(joint, actuatedJoints = []):
    joint_actuated = 0
    for aj in actuatedJoints:
        if( aj in joint.name ):
            joint_actuated = 1
    return joint_actuated

class EdgeData:
    def __init__(self, joint, joint1, joint2, actuatedJoints = []):
        # JointTypeDict = {'fixed':0, 'revolute':1, 'prismatic':2, 'continuous':3}
        # actuatedJoints = ['coxa', 'femur', 'tibia']

        self.joint_actuated = check_actuated(joint, actuatedJoints)
        self.axis = joint.axis
        
        T1 = get_diff_T(joint.origin, joint1.origin)
        T2 = get_diff_T(joint.origin, joint2.origin)

        ori1, pos1 = zyx_p_from_T(T1)
        ori2, pos2 = zyx_p_from_T(T2)

        self.data = [self.joint_actuated]
        self.data.extend(self.axis)
        self.data.extend(pos1)
        self.data.extend(ori1)
        self.data.extend(pos2)
        self.data.extend(ori2)

        # print(joint1.name + '- <' + joint.name + '> -' + joint2.name )
        # print(self.data)

class JointData:
    def __init__(self, joint, actuatedJoints = []):
        JointTypeDict = {'fixed':0, 'revolute':1, 'prismatic':2, 'continuous':3}
        # actuatedJoints = ['coxa', 'femur', 'tibia']

        self.joint_type = joint.joint_type
        self.joint_type_idx = JointTypeDict[self.joint_type]

        self.joint_actuated = 0
        for aj in actuatedJoints:
            if( aj in joint.name ):
                self.joint_actuated = 1

        self.parent = joint.parent
        self.child = joint.child


        r = R.from_matrix(joint.origin[0:3,0:3])
        self.ori = r.as_euler('zyx', degrees=False)
        
        # axis, angle = axis_angle_from_rotation(joint.origin[0:3,0:3])
        # self.ori_axis = axis.tolist()
        # self.ori_angle = angle
        # self.ori = axis.tolist()
        # self.ori.append(angle)

        self.pos = joint.origin[0:3,3].tolist()
        self.axis = joint.axis

        self.data = [self.joint_type_idx]
        self.data.append(self.joint_actuated)
        self.data.extend(self.axis)
        self.data.extend(self.pos)
        self.data.extend(self.ori)

        # print(self.data)

class LinkData:
    def __init__(self, inertial):
        self.mass = inertial.mass
        self.inertia = [ inertial.inertia[0,0], inertial.inertia[1,1], 
                    inertial.inertia[2,2], inertial.inertia[0,1],  
                    inertial.inertia[1,2], inertial.inertia[0,2] ]
                    # Ixx, Iyy, Izz, Ixy, Iyz, Izx
        
        # axis, angle = axis_angle_from_rotation(inertial.origin[0:3,0:3])
        # self.ori_axis = axis.tolist()
        # self.ori_angle = angle
        # self.ori = axis.tolist()
        # self.ori.append(angle)
        self.pos = inertial.origin[0:3,3].tolist()

        r = R.from_matrix(inertial.origin[0:3,0:3])
        self.ori = r.as_euler('zyx', degrees=False)    

        self.data = [self.mass]
        self.data.extend(self.inertia)
        self.data.extend(self.pos)
        self.data.extend(self.ori)

        # print(self.data)

def loadRobotURDF(fn):
    return URDF.load(fn)


# from magnetoDefinition import *
# import os

# CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# FootNames = ['base_link','AR','BR','BL','AL']
# ActuatedJoints = ['coxa', 'femur', 'tibia']
# robot = loadRobotURDF(CURRENT_DIR_PATH + '/MagnetoSim_Dart.urdf')

# nodes = []
# masseps = 1e-5
# for link in robot.links:
#     for fn in FootNames:
#         #print(fn)
#         if( fn in link.name and link.inertial.mass > masseps ):
#             print(link.name)
#             nodes.append(LinkData(link.inertial))
#             MagnetoGraphNode[link.name ] = len(nodes)

# edges = []
# srcs = [] 
# dest = []


# joints = []
# srcsj = [] 
# destj = []

# for joint in robot.joints:
#     for fn in FootNames:
#         #print(fn)
#         if( fn in joint.name and joint.joint_type != 'fixed'):
#             print(joint.name)
#             # edges.append(JointData(joint, ActuatedJoints))
#             # edges.append(EdgeData(joint, joint1, joint2, ActuatedJoints))

#             joints.append(joint)            
#             srcsj.append(MagnetoGraphNode[joint.parent])
#             destj.append(MagnetoGraphNode[joint.child])

# # parent : srcs



# for link_idx in range(len(nodes)):
#     # i : list of joint where the parent link == link_idx
#     idxs = [i for i, x in enumerate(srcsj) if x==link_idx]

#     if(len(idxs) > 1):
#         idxs1 = shuffle_list(idxs)
#         idxs2 = shuffle_list_reverse(idxs)

#         # print(idxs)
#         # print(idxs1)
#         # print(idxs2)

#         for j, j1, j2 in zip( idxs, idxs1, idxs2 ):
#             joint = joints[j]
#             joint1 = joints[j1]
#             joint2 = joints[j2]
#             edges.append(EdgeData(joint, joint1, joint2, ActuatedJoints))
#             srcs.append(MagnetoGraphNode[joint.parent])
#             dest.append(MagnetoGraphNode[joint.child])
    
#     elif(len(idxs) == 1):
#         idxs1 = [i for i, x in enumerate(destj) if x==link_idx]
    
#         # print(idxs)
#         # print(idxs1)

#         for j, j1 in zip( idxs, idxs1 ):
#             joint = joints[j]
#             joint1 = joints[j1]
#             edges.append(EdgeData(joint, joint1, joint1, ActuatedJoints))
#             srcs.append(MagnetoGraphNode[joint.parent])
#             dest.append(MagnetoGraphNode[joint.child])


# print('done!')
# print('node='+str(len(nodes)))
# print('edge='+str(len(edges)))
# print('dest='+str(len(dest)))
# print('srcs='+str(len(srcs)))
