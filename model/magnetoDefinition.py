# magnetoDefinition.py
MagnetoLink = {
    "basePosX" : 0,
    "basePosY" : 1,
    "basePosZ" : 2,
    "baseRotZ" : 3,
    "baseRotY" : 4,
    "base_link" : 5,
    "AL_coxa_link" : 6,
    "AL_femur_link" : 7,
    "AL_tibia_link" : 8,
    "AL_depth_sensor_link" : 9,
    "gazebo_AL_leg_optical_frame" : 10,
    "AL_foot_link_1" : 11,
    "AL_foot_link_2" : 12,
    "AL_foot_link_3" : 13,
    "AL_foot_link" : 14,
    "AR_coxa_link" : 15,
    "AR_femur_link" : 16,
    "AR_tibia_link" : 17,
    "AR_depth_sensor_link" : 18,
    "gazebo_AR_leg_optical_frame" : 19,
    "AR_foot_link_1" : 20,
    "AR_foot_link_2" : 21,
    "AR_foot_link_3" : 22,
    "AR_foot_link" : 23,
    "BL_coxa_link" : 24,
    "BL_femur_link" : 25,
    "BL_tibia_link" : 26,
    "BL_depth_sensor_link" : 27,
    "gazebo_BL_leg_optical_frame" : 28,
    "BL_foot_link_1" : 29,
    "BL_foot_link_2" : 30,
    "BL_foot_link_3" : 31,
    "BL_foot_link" : 32,
    "BR_coxa_link" : 33,
    "BR_femur_link" : 34,
    "BR_tibia_link" : 35,
    "BR_depth_sensor_link" : 36,
    "gazebo_BR_leg_optical_frame" : 37,
    "BR_foot_link_1" : 38,
    "BR_foot_link_2" : 39,
    "BR_foot_link_3" : 40,
    "BR_foot_link" : 41,
    "front_indicator" : 42
}

MagnetoJoint = {
    "basePosX" : 0,
    "basePosY" : 1,
    "basePosZ" : 2,
    "baseRotZ" : 3,
    "baseRotY" : 4,
    "_base_joint" : 5,
    "AL_coxa_joint" : 6,
    "AL_femur_joint" : 7,
    "AL_tibia_joint" : 8,
    "AL_foot_joint_1" : 9,
    "AL_foot_joint_2" : 10,
    "AL_foot_joint_3" : 11,
    "AR_coxa_joint" : 12,
    "AR_femur_joint" : 13,
    "AR_tibia_joint" : 14,
    "AR_foot_joint_1" : 15,
    "AR_foot_joint_2" : 16,
    "AR_foot_joint_3" : 17,
    "BL_coxa_joint" : 18,
    "BL_femur_joint" : 19,
    "BL_tibia_joint" : 20,
    "BL_foot_joint_1" : 21,
    "BL_foot_joint_2" : 22,
    "BL_foot_joint_3" : 23,
    "BR_coxa_joint" : 24,
    "BR_femur_joint" : 25,
    "BR_tibia_joint" : 26,
    "BR_foot_joint_1" : 27,
    "BR_foot_joint_2" : 28,
    "BR_foot_joint_3" : 29
}


MagnetoGraphNode = {'base_link': 0, 
                    'AR_coxa_link': 1, 'AR_femur_link': 2, 'AR_tibia_link': 3, 
                    'AR_foot_link_1': 4, 'AR_foot_link_2': 5, 'AR_foot_link_3': 6, 'AR_depth_sensor_link': 7, 
                    'BR_coxa_link': 8, 'BR_femur_link': 9, 'BR_tibia_link': 10, 
                    'BR_foot_link_1': 11, 'BR_foot_link_2': 12, 'BR_foot_link_3': 13, 'BR_depth_sensor_link': 14, 
                    'BL_coxa_link': 15, 'BL_femur_link': 16, 'BL_tibia_link': 17, 
                    'BL_foot_link_1': 18, 'BL_foot_link_2': 19, 'BL_foot_link_3': 20, 'BL_depth_sensor_link': 21, 
                    'AL_coxa_link': 22, 'AL_femur_link': 23, 'AL_tibia_link': 24, 
                    'AL_foot_link_1': 25, 'AL_foot_link_2': 26, 'AL_foot_link_3': 27, 'AL_depth_sensor_link': 28}

MagnetoGraphEdge = {'AR_coxa_joint': 0, 'AR_femur_joint': 1, 'AR_tibia_joint': 2, 
                    'AR_foot_joint_1': 3, 'AR_foot_joint_2': 4, 'AR_foot_joint_3': 5, 
                    'BR_coxa_joint': 6, 'BR_femur_joint': 7, 'BR_tibia_joint': 8, 
                    'BR_foot_joint_1': 9, 'BR_foot_joint_2': 10, 'BR_foot_joint_3': 11, 
                    'BL_coxa_joint': 12, 'BL_femur_joint': 13, 'BL_tibia_joint': 14, 
                    'BL_foot_joint_1': 15, 'BL_foot_joint_2': 16, 'BL_foot_joint_3': 17, 
                    'AL_coxa_joint': 18, 'AL_femur_joint': 19, 'AL_tibia_joint': 20, 
                    'AL_foot_joint_1': 21, 'AL_foot_joint_2': 22, 'AL_foot_joint_3': 23}

MagnetoGraphEdgeSender = {'AR_coxa_joint': 0, 'AR_femur_joint': 1, 'AR_tibia_joint': 2, 
                        'AR_foot_joint_1': 3, 'AR_foot_joint_2': 4, 'AR_foot_joint_3': 5, 
                        'BR_coxa_joint': 0, 'BR_femur_joint': 8, 'BR_tibia_joint': 9, 
                        'BR_foot_joint_1': 10, 'BR_foot_joint_2': 11, 'BR_foot_joint_3': 12, 
                        'BL_coxa_joint': 0, 'BL_femur_joint': 15, 'BL_tibia_joint': 16, 
                        'BL_foot_joint_1': 17, 'BL_foot_joint_2': 18, 'BL_foot_joint_3': 19, 
                        'AL_coxa_joint': 0, 'AL_femur_joint': 22, 'AL_tibia_joint': 23, 
                        'AL_foot_joint_1': 24, 'AL_foot_joint_2': 25, 'AL_foot_joint_3': 26}

MagnetoGraphEdgeReceiver = {'AR_coxa_joint': 1, 'AR_femur_joint': 2, 'AR_tibia_joint': 3, 
                            'AR_foot_joint_1': 4, 'AR_foot_joint_2': 5, 'AR_foot_joint_3': 6, 
                            'BR_coxa_joint': 8, 'BR_femur_joint': 9, 'BR_tibia_joint': 10, 
                            'BR_foot_joint_1': 11, 'BR_foot_joint_2': 12, 'BR_foot_joint_3': 13, 
                            'BL_coxa_joint': 15, 'BL_femur_joint': 16, 'BL_tibia_joint': 17, 
                            'BL_foot_joint_1': 18, 'BL_foot_joint_2': 19, 'BL_foot_joint_3': 20, 
                            'AL_coxa_joint': 22, 'AL_femur_joint': 23, 'AL_tibia_joint': 24, 
                            'AL_foot_joint_1': 25, 'AL_foot_joint_2': 26, 'AL_foot_joint_3': 27}
