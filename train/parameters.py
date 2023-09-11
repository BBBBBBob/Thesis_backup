import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extrinsic_1 = torch.tensor(np.concatenate((np.diag([-1, -1, 1]), np.array([0, -0.05, 0]).reshape(3, 1)), axis=1),
                           dtype=torch.float32, device=device)
extrinsic_2 = torch.tensor(np.concatenate((np.diag([-1, -1, 1]), np.array([0, 0.05, 0]).reshape(3, 1)), axis=1),
                           dtype=torch.float32, device=device)

K = torch.tensor(np.array([-0.008142077171285192,
                           0.04871829310795375,
                           -0.04965037414851462,
                           0.010537777543667439]), dtype=torch.float32, device=device)

intrinsic = torch.tensor(np.array([[287.92069652603794, 0, 433.65166811563813],
                                   [0, 287.87322535974465, 405.40096052696083],
                                   [0, 0, 1]]), dtype=torch.float32, device=device)

optical_center = torch.tensor([[intrinsic[0, 2]], [intrinsic[1, 2]]], dtype=torch.float32, device=device)
focal_length = (intrinsic[0, 0] + intrinsic[1, 1]) / 2

normalize = torch.tensor([[848], [800]], dtype=torch.float32, device=device)

left_cam_pos = torch.tensor([0, -0.05, 0], dtype=torch.float32, device=device)
right_cam_pos = torch.tensor([0, 0.05, 0], dtype=torch.float32, device=device)