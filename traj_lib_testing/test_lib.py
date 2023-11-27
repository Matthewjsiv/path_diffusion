import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import matplotlib.cm as cm
cmap = cm.viridis

TRAJ_LIB = np.load('traj_lib.npy')#[:,:,:2]#[::50]
N_TRAJ = len(TRAJ_LIB)
print(TRAJ_LIB.shape)
# Z_APPEND = np.zeros((TRAJ_LIB.shape[0],1))

# TRAJ_LIB = np.hstack([TRAJ_LIB, Z_APPEND])

def transformed_lib(pose):

    lib = TRAJ_LIB[:,:,:2].copy()

    submat = pose[:3, :3]
    yaw = -np.arctan2(submat[1, 0], submat[0, 0]) + np.pi/2

    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                [np.sin(yaw), np.cos(yaw)]])

    points = lib.reshape(-1,2)

    rotated_points = points @ rotation_matrix.T

    points = rotated_points.reshape(N_TRAJ,-1,2)

    return points

def pose_msg_to_se3(msg):
        # quaternion_msg = msg[3:7]
        #msg is in xyzw
        Q = np.array([msg[6], msg[3], msg[4], msg[5]])
        rot_mat = quaternion_rotation_matrix(Q)

        se3 = np.zeros((4, 4))
        se3[:3, :3] = rot_mat
        se3[0, 3] = msg[0]
        se3[1, 3] = msg[1]
        se3[2, 3] = msg[2]
        se3[3, 3] = 1

        return se3

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix. Copied from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
            This rotation matrix converts a point in the local reference
            frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])

    return rot_matrix


dataset = torch.load('context_mppi_pipe_1.pt')
# print(dataset['observation']['state'].shape)
# traj = dataset['observation']['state']

# print(dataset['observation'].keys())

length = dataset['observation']['state'].shape[0]
obs = dataset['observation']
print(obs.keys())
for i in range(50,length):
    #NOTE: data is already reversed
    #in general, sam already covered the preprocessing so we need to do it at runtime but not here
    costmap = obs['local_costmap_data'][i][0]
    odom = obs['state'][i]

    nx = obs['local_costmap_height'][i]
    ny = obs['local_costmap_width'][i]
    res = obs['local_costmap_resolution'][i][0].item() #assuming square
    print(res)
    origin = obs['local_costmap_origin']

    pose_se3 = pose_msg_to_se3(odom)

    #THIS SHOULD EVENTUALLY REPLACED WITH A MORE PRINCIPLED WAY
    trajs = transformed_lib(pose_se3)
    # trajs[:,:,0] += int(costmap.shape[0]/2)
    # trajs[:,:,1] += int(costmap.shape[1]/2)

    #add origin + half length/width
    #then world to grid

    #for viz
    costmap -= costmap.min()
    costmap /= costmap.max()
    costmap = costmap.numpy()

    # trajs_disc = trajs.astype(int)
    # trajs_disc /= res
    # trajs_disc[:,:,0] += costmap.shape[0]/2
    # trajs_disc[:,:,1] += costmap.shape[1]/2

    trajs_disc = ((trajs - np.array([-30., -30]).reshape(1, 1, 2)) / res).astype(np.int32)

    # import pdb;pdb.set_trace()

    # print(trajs.dtype)
    # costmap[trajs_disc[:,:,0],trajs_disc[:,:,1]] = 1
    # print(np.min(costmap), np.max(costmap))
    costs = costmap[trajs_disc[:,:,0],trajs_disc[:,:,1]].sum(axis=1)
    # print(costs.shape)
    costs /= costs.max()

    plt.imshow(costmap,origin='lower',extent=[-30, 30, -30, 30])
    for i in range(N_TRAJ):
        plt.plot(trajs[i,:,1],trajs[i,:,0],c=cmap(costs[i]))
    plt.show()

    ids = np.argsort(costs)[:100]
    # for i in ids:
    #     plt.plot(trajs[i,:,1],trajs[i,:,0],c=cmap(costs[i]))
    # plt.show()
    # costmap[trajs_disc[ids,:,0],trajs_disc[ids,:,1]] = 1
    # cv2.imshow('test',costmap)
    # cv2.waitKey(1)

    # if i > 50:
    #     break
