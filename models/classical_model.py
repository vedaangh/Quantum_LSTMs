import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import torch
import torch.nn as nn
device = torch.device("cuda")


# initialising key constants

INPUT_DIM = 1  # num features per input
OUTPUT_DIM = 1
HIDDEN_DIM = 5
vt_dim = INPUT_DIM + HIDDEN_DIM

DEPTH = 2



class LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.LSTM_l = nn.LSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, batch_first=True)
        self.y_l = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, input_seq):
        y_seq = []
        batch_size, seq_length, feature_size = input_seq.size()  # shape is  seq x features assume no batches

        ct = torch.zeros(1, batch_size, HIDDEN_DIM)  # .to(device)
        ht = torch.zeros(1, batch_size, HIDDEN_DIM)  # .to(device)
        h_out, (ht, ct) = self.LSTM_l(input_seq, (ht, ct))

        y = self.y_l(ht[0])
        return y


print(f"number of parameters in classical model: {sum(p.numel() for p in LSTM().parameters() if p.requires_grad)}")