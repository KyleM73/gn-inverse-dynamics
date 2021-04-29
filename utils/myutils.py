import math
import numpy as np
from scipy import spatial

def quat_to_rot(quat):
  # input arg quat : w,x,y,z
  # output rotation matrix : 3x3 array
  r = spatial.transform.Rotation.from_quat(quat)
  return r

def quat_to_rot_axis(quat, axis):
  # quat : w,x,y,z
  # axis : 'x','y','z'
  axis_dict = {'x':0,'y':1,'z':2}
  r = quat_to_rot(quat).as_matrix()
  # print(r)
  rz = r[0:3, axis_dict[axis]]
  return rz


def normalize(arr):
    "arr : array"
    output_arr = []
    array_norm = 0
    for element in arr:
        array_norm += element**2

    if array_norm > 1e-3:
        for element in arr:
            output_arr.append(element / array_norm)

    return output_arr

def cwise_product(arr1, arr2):    
    if(len(arr1) == len(arr2)):
        output = []
        for a1, a2 in zip(arr1, arr2):
            output.append(a1*a2)
        return output
    else:
        raise ValueError('input array size is wrong')

def angle_between_axes(axis1, axis2):
    EPS = 1e-3
    if(np.linalg.norm(axis1) < EPS or np.linalg.norm(axis2) < EPS  ):
        raise ValueError('input array size is wrong')

    cos_theta = np.dot(axis1, axis2) / np.linalg.norm(axis1) / np.linalg.norm(axis2)
  
    if(cos_theta>1.0-EPS):
        return 0
    elif(cos_theta<-1.0+EPS):
        return np.pi
    else:
        return math.acos(cos_theta)


def mod_angle(angle, minmax):
    mod_min = minmax[0]
    mod_max = minmax[1]
    mod_size = mod_min - mod_max
    
    return np.remainder(angle - mod_min, mod_size)  + mod_min