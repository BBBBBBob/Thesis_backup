import numpy as np
from scipy.optimize import linear_sum_assignment
from train.net import GRU_ATT
from parameters import *

model_path = '../weight/best_model_evolve_enc.pt'
net = GRU_ATT(hidden_size=hidden_size, input_size=input_size,
              output_size=output_size, output_len=output_len,
              num_layers=num_layers, mode=8, temp=1, dropout=0).to(device=device)