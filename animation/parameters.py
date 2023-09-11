import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_len = 8
output_len = 16
hidden_size = 64
input_size = 4
output_size = 3
num_layers = 1

## initialize Kalman filter
dt = 1/30

K = np.array([-0.008142077171285192,
              0.04871829310795375,
              -0.04965037414851462,
              0.010537777543667439])

extrinsic_1 = np.concatenate((np.diag([-1, -1, 1]), np.array([0, -0.05, 0]).reshape(3, 1)), axis=1)
extrinsic_2 = np.concatenate((np.diag([-1, -1, 1]), np.array([0, 0.05, 0]).reshape(3, 1)), axis=1)
# extrinsic_1 = np.concatenate((np.diag([-1, -1, 1]), np.array([-0.05, 0, 0]).reshape(3, 1)), axis=1)
# extrinsic_2 = np.concatenate((np.diag([-1, -1, 1]), np.array([0.05, 0, 0]).reshape(3, 1)), axis=1)

intrinsic = np.array([[287.92069652603794, 0, 433.65166811563813],
                      [0, 287.87322535974465, 405.40096052696083],
                      [0, 0, 1]])

optical_center = np.array([intrinsic[0, 2], intrinsic[1, 2]])
focal_length = (intrinsic[0, 0] + intrinsic[1, 1]) / 2

left_cam_pos = np.array([0, -0.05, 0])
right_cam_pos = np.array([0, 0.05, 0])
baseline = np.array([0, -0.1, 0])

# left_cam_pos = np.array([-0.23, 0, -0.055])
# right_cam_pos = np.array([0.37, 0, -0.055])
# baseline = np.array([0.1, 0, 0])

K_th = torch.tensor(K, dtype=torch.float32, device=device)
extrinsic_1_th = torch.tensor(extrinsic_1, dtype=torch.float32, device=device)
extrinsic_2_th = torch.tensor(extrinsic_2, dtype=torch.float32, device=device)
intrinsic_th = torch.tensor(intrinsic, dtype=torch.float32, device=device)
optical_center_th = torch.tensor(np.array([[intrinsic[0, 2]], [intrinsic[1, 2]]]), dtype=torch.float32, device=device)
focal_length_th = torch.tensor(focal_length, dtype=torch.float32, device=device)
normalize_th = torch.tensor([[848], [800]], dtype=torch.float32, device=device)