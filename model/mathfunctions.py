import numpy as np
from scipy.spatial.transform import Rotation


def get_diff_T(T1,T2):
    T = inv_T(T1)*T2
    return T

def inv_T(T):
    R = T[0:3, 0:3]
    p = T[0:3,3].tolist()

    invR = inv_R(R)
    invp = - np.matmul(invR, p)

    return T_from_R_p(invR, invp)

def inv_R(R):
    invR = np.transpose(R)
    return invR

def T_from_R_p(R,p):
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3] = p
    return T

def R_p_from_T(T):
    R = T[0:3, 0:3]
    p = T[0:3,3].tolist()
    return R,p

def zyx_p_from_T(T):
    R, p = R_p_from_T(T)
    
    rot = Rotation.from_matrix(R)
    zyx = rot.as_euler('zyx', degrees=False)

    return zyx, p

