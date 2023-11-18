import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pickle
import random

from utils import compute_single_cls_ap


class EnvDataset(Dataset):
    """
    Returns:
        traj1 states: (B, T, 3, 192, 256)
        traj1 actions: (B, T)
        traj1 ap: (B,)
        traj2 states: (B, T, 3, 192, 256)
        traj2 actions: (B, T)
        traj2 ap: (B,)
    """
    def __init__(self, data_dir):
        self.traj_paths = [os.path.join(data_dir, traj_dir) for traj_dir in os.listdir(data_dir)]

    def process_one_traj(self, traj_path, part_len=13):
        traj_len = len(os.listdir(traj_path))
        # 截取同样长度的trajectory
        traj_start_idx = np.random.randint(0, traj_len-part_len+1)
        states = np.zeros((part_len, 3, 192, 256))
        actions = np.zeros((part_len, 4))
        detections = []
        labels = []
        for idx in range(part_len):
            sample_path = os.path.join(traj_path, f"sample{traj_start_idx+idx}.pkl")
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)
            states[idx] = sample[0]
            actions[idx][sample[1]] = 1
            detections.append(sample[2])
            labels.append(sample[3])
        ap = compute_single_cls_ap(detections, labels, iou_thres=0.5)
        return states, actions, ap

    def __getitem__(self, index):
        traj_path1 = self.traj_paths[index%len(self)]
        # 样本1
        states1, actions1, ap1 = self.process_one_traj(traj_path1)
        # 构造对比样本2
        compare_list = list(range(0, index)) + list(range(index+1, len(self)))
        traj_path2_idx = random.choice(compare_list)
        traj_path2 = self.traj_paths[traj_path2_idx%len(self)]
        state2, actions2, ap2 = self.process_one_traj(traj_path2)
        return torch.from_numpy(states1).float(), torch.from_numpy(actions1).float(), torch.tensor(ap1).float(), \
            torch.from_numpy(state2).float(), torch.from_numpy(actions2).float(), torch.tensor(ap2).float()

    def __len__(self):
        return len(self.traj_paths)


if __name__ == '__main__':
    batch_size = 4
    shuffle = True
    num_workers = 4
    dataset = EnvDataset('/home/agent/Code/ackermann_car_nav/data/trajectories/policy_my')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    for states1, actions1, ap1, states2, actions2, ap2 in dataloader:
        print(states1.size(), actions1.size(), ap1.size(), ap1.dtype)
