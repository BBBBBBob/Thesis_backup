import torch
import re
import glob
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np

class dataloader(Dataset):
    def __init__(self, camera_path_1, camera_path_2):
        self.files_path_1 = sorted(glob.glob(camera_path_1 + '/*'),
                                   key=lambda s: int(re.findall(r'\d+', s)[1] + re.findall(r'\d+', s)[2]))
        self.files_path_2 = sorted(glob.glob(camera_path_2 + '/*'),
                                   key=lambda s: int(re.findall(r'\d+', s)[1] + re.findall(r'\d+', s)[2]))
        assert len(self.files_path_1) == len(self.files_path_2), 'Amount of data from two cameras should be equal!'

        self.history_rot_pix_1_list = []
        self.history_rot_pix_2_list = []
        self.history_pos_pix_1_list = []
        self.history_pos_pix_2_list = []
        self.future_pos_pix_1_list = []
        self.future_pos_pix_2_list = []
        self.future_pos_pix_undist_1_list = []
        self.future_pos_pix_undist_2_list = []
        self.future_diff_world_list = []
        self.history_pos_world_list = []

        for i in range(len(self.files_path_1)):
            file_1 = open(self.files_path_1[i], 'rb')
            data_1 = pickle.load(file_1)
            file_2 = open(self.files_path_2[i], 'rb')
            data_2 = pickle.load(file_2)
            self.history_rot_pix_1_list.append(data_1['history_rot_pix'])
            self.history_rot_pix_2_list.append(data_2['history_rot_pix'])
            self.history_pos_pix_1_list.append(data_1['history_pos_pix'])
            self.history_pos_pix_2_list.append(data_2['history_pos_pix'])
            self.future_pos_pix_1_list.append(data_1['future_pos_pix'])
            self.future_pos_pix_2_list.append(data_2['future_pos_pix'])
            self.future_pos_pix_undist_1_list.append(data_1['future_pos_pix_undist'])
            self.future_pos_pix_undist_2_list.append(data_2['future_pos_pix_undist'])
            self.future_diff_world_list.append(data_1['future_diff_world'])
            self.history_pos_world_list.append(data_1['history_pos_world'])

    def __len__(self):
        return len(self.files_path_1)

    def __getitem__(self, idx):
        history_rot_pix_1 = torch.from_numpy(self.history_rot_pix_1_list[idx]).to(dtype=torch.float32)
        history_rot_pix_2 = torch.from_numpy(self.history_rot_pix_2_list[idx]).to(dtype=torch.float32)
        history_rot_pix = torch.concat((history_rot_pix_1, history_rot_pix_2), dim=1)

        future_diff_world = torch.from_numpy(self.future_diff_world_list[idx]).to(dtype=torch.float32)

        history_pos_pix_1 = torch.from_numpy(self.history_pos_pix_1_list[idx]).to(dtype=torch.float32)
        history_pos_pix_2 = torch.from_numpy(self.history_pos_pix_2_list[idx]).to(dtype=torch.float32)

        future_pos_pix_1 = torch.from_numpy(self.future_pos_pix_1_list[idx]).to(dtype=torch.float32)
        future_pos_pix_2 = torch.from_numpy(self.future_pos_pix_2_list[idx]).to(dtype=torch.float32)

        future_pos_pix_undist_1 = torch.from_numpy(self.future_pos_pix_undist_1_list[idx]).to(dtype=torch.float32)
        future_pos_pix_undist_2 = torch.from_numpy(self.future_pos_pix_undist_2_list[idx]).to(dtype=torch.float32)

        history_pos_world = torch.from_numpy(self.history_pos_world_list[idx]).to(dtype=torch.float32)

        return history_rot_pix, future_diff_world, history_pos_pix_1, history_pos_pix_2, future_pos_pix_1, future_pos_pix_2, future_pos_pix_undist_1, \
            future_pos_pix_undist_2, history_pos_world


