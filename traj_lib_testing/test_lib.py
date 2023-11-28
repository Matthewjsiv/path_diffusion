#torch imports
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

#python imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
cmap = cm.viridis

TEST_LOCALLY = False

TRAJ_LIB = np.load('traj_lib.npy')#[:,:,:2]#[::50]
N_TRAJ = len(TRAJ_LIB)
print(TRAJ_LIB.shape)

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


class TrajLib(object):
    def __init__(self, dataset_file, top_k, cost_threshold, num_buckets=3, visualize=False):
        self.visualize = visualize
        self.dataset = torch.load(dataset_file)
        print("dataset loaded")
        self.obs = self.dataset['observation']
        self.cost_threshold = cost_threshold
        self.num_buckets = num_buckets
        self.top_k = top_k

    #preprocesses data at a given time t and returns cost and trajectories
    def preprocess_data(self, t):
        costmap = self.obs['local_costmap_data'][t][0].numpy()
        odom = self.obs['state'][t]
        res = self.obs['local_costmap_resolution'][t][0].item() #assuming square
        pose_se3 = pose_msg_to_se3(odom)
        #THIS SHOULD EVENTUALLY REPLACED WITH A MORE PRINCIPLED WAY
        trajs = transformed_lib(pose_se3)
        trajs_disc = ((trajs - np.array([-30., -30]).reshape(1, 1, 2)) / res).astype(np.int32)
        costs = costmap[trajs_disc[:,:,0],trajs_disc[:,:,1]].sum(axis=1)
        #keep trajectories that are below a certain cost
        idx_below_cost_thresh = np.where(costs < self.cost_threshold)[0]
        costs = costs[idx_below_cost_thresh]
        trajs = trajs[idx_below_cost_thresh]
        return trajs, costs, costmap

    def get_top_trajs(self, t):
        #preprocess data
        trajs, costs, costmap = self.preprocess_data(t)
        #split trajs and costs into bucket
        split_traj_lib = np.array_split(trajs, self.num_buckets, axis=0)
        split_costs = np.array_split(costs, self.num_buckets)
        #iterate through buckets and extract top k/num elements elements --> assume perfect division for now
        top_k_per_bucket = self.top_k // self.num_buckets
        best_split_trajs, best_split_costs = [], []
        for idx in range(len(split_traj_lib)):
            least_cost_bucket_idx = np.argsort(split_costs[idx])[:top_k_per_bucket].astype(np.int32)
            least_cost_trajs = split_traj_lib[idx][least_cost_bucket_idx]
            least_costs = split_costs[idx][least_cost_bucket_idx]
            best_split_trajs.append(least_cost_trajs)
            best_split_costs.append(least_costs)
        
        return np.concatenate(best_split_trajs), np.concatenate(best_split_costs), costmap

class TrajLibDataset(Dataset):
    def __init__(self, dataset_file, top_k, cost_threshold, num_buckets=3):
        self.traj_lib = TrajLib(dataset_file=dataset_file, top_k=top_k, cost_threshold=cost_threshold, num_buckets=num_buckets)
    
    def __len__(self):
        return self.traj_lib.dataset['observation']['state'].shape[0]

    def __getitem__(self, idx):
        #get top trajs from trajectory library
        best_trajs, best_costs, costmap = self.traj_lib.get_top_trajs(idx)
        return torch.tensor(costmap), torch.tensor(best_trajs)
    

if __name__ == "__main__":
    if TEST_LOCALLY:
        traj_lib = TrajLib(dataset_file='context_mppi_pipe_1.pt', top_k=30, cost_threshold=18.5, visualize=True)
        for i in range(350,500):
            best_trajs, best_costs, costmap = traj_lib.get_top_trajs(i)
            #normalize for vizualization
            costmap -= costmap.min()
            costmap /= costmap.max()
            best_costs /= best_costs.max()

            #plot top trajs on costmap (image per timestep)
            plt.imshow(costmap,origin='lower',extent=[-30, 30, -30, 30])
            for idx in range(30):
                plt.plot(best_trajs[idx,:,1], best_trajs[idx,:,0], c=cmap(best_costs[idx]))
            plt.show()


if __name__ == "__main__":

    traj_lib = TrajLib(dataset_file='context_mppi_pipe_1.pt', top_k=30, cost_threshold=18.5, visualize=True)
    for i in range(350,351):
        best_trajs, best_costs, costmap = traj_lib.get_top_trajs(i)


