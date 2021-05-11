from urdfpy import URDF
from geometry import axis_angle_from_rotation

# links, joints, actuated_joints

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

        axis, angle = axis_angle_from_rotation(joint.origin[0:3,0:3])
        self.ori_axis = axis.tolist()
        self.ori_angle = angle
        self.pos = joint.origin[0:3,3].tolist()

        self.axis = joint.axis

        self.data = [self.joint_type_idx]
        self.data.append(self.joint_actuated)
        self.data.extend(self.axis)
        self.data.extend(self.pos)
        self.data.extend(self.ori_axis)
        self.data.append(self.ori_angle)

        #print(self.data)

class LinkData:
    def __init__(self, inertial):
        self.mass = inertial.mass
        self.inertia = [ inertial.inertia[0,0], inertial.inertia[1,1], 
                    inertial.inertia[2,2], inertial.inertia[0,1],  
                    inertial.inertia[1,2], inertial.inertia[0,2] ]
                    # Ixx, Iyy, Izz, Ixy, Iyz, Izx
        axis, angle = axis_angle_from_rotation(inertial.origin[0:3,0:3])
        self.ori_axis = axis.tolist()
        self.ori_angle = angle
        self.pos = inertial.origin[0:3,3].tolist()        

        self.data = [self.mass]
        self.data.extend(self.inertia)
        self.data.extend(self.pos)
        self.data.extend(self.ori_axis)
        self.data.append(self.ori_angle)

        #print(self.data)

def loadRobotURDF(fn):
    return URDF.load(fn)



'''
import os

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

robot = loadRobotURDF(CURRENT_DIR_PATH+'/model/MagnetoSim_Dart.urdf')
CURRENT_DIR_PAT = 
nodes = []
masseps = 1e-5
for link in robot.links:
    for fn in FootNames:
        #print(fn)
        if( fn in link.name and link.inertial.mass > masseps ):
            print(link.name)
            nodes.append(LinkData(link.inertial))
            NodeIdDict[link.name ] = len(nodes)

edges = []
srcs = [] 
dest = []

for joint in robot.joints:
    for fn in FootNames:
        #print(fn)
        if( fn in joint.name and joint.joint_type != 'fixed'):
            print(joint.name)
            edges.append(JointData(joint))
            srcs.append(NodeIdDict[joint.parent])
            dest.append(NodeIdDict[joint.child])

print('done!')
print('node='+str(len(nodes)))
print('edge='+str(len(edges)))
print('dest='+str(len(dest)))
print('srcs='+str(len(srcs)))
'''
