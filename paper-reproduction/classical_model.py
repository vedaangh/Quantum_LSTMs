import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import torch
import torch.nn as nn
device = torch.device("cuda")
INPUT_DIM = 1
OUTPUT_DIM = 1
HIDDEN_DIM = 3
VT_DIM = INPUT_DIM+HIDDEN_DIM

DEPTH = 2



class QLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.forget_l = nn.Linear(VT_DIM, OUTPUT_DIM)
        self.input_l = nn.Linear(VT_DIM, OUTPUT_DIM)
        self.update_l = nn.Linear(VT_DIM, OUTPUT_DIM)
        self.output_l = nn.Linear(VT_DIM, OUTPUT_DIM)
        self.y_l = nn.Linear(VT_DIM, OUTPUT_DIM)
        self.ht_l = nn.Linear(VT_DIM, HIDDEN_DIM)
    def forward(self, input_seq):
        y_seq = []
        batch_size, seq_length, feature_size = input_seq.size()  # shape is  seq x features assume no batches

        ct = torch.zeros(batch_size, VT_DIM)  # .to(device)
        ht = torch.zeros(batch_size, HIDDEN_DIM)  # .to(device)

        for t in range(seq_length):
            x = input_seq[:, t, :]  # take all features input at timestep t

            vt = torch.concat((x, ht), dim=1)

            ft = torch.sigmoid(self.forget_l(vt))  # forget
            it = torch.sigmoid(self.input_l(vt))  # input
            gt = torch.tanh(self.update_l(vt))  # update
            ot = torch.sigmoid(self.output_l(vt))  # output

            ct = ft * ct + (it * gt)
            ht = self.ht_l(ot * torch.tanh(ct))
            yt = self.y_l(ot * torch.tanh(ct))
            y_seq.append(yt)
        y = y_seq[-1] # we are predicting T_n+1 | T_1->n

        return y