import torch
import matplotlib.pyplot as plt
import numpy as np


dataset = torch.load('context_mppi_pipe_0.pt')
# print(dataset['observation']['state'].shape)
traj = dataset['observation']['state']
plt.plot(traj[:,0],traj[:,1])
plt.show()
# print(dataset['observation']['local_costmap_data'].shape)

costmaplist = dataset['observation']['local_costmap_data']

costmap1 = costmaplist[1100][0]

plt.imshow(costmap1)
plt.show()

# dataset = torch.load('/home/matthew/racer_data/clean_torch_train_h75/traj_17229.pt')
# print(dataset.keys())
