# import dependencies
import math
import torch
import matplotlib.pyplot as plt
import data_gen as data
import torch.nn as nn
import random

# import models
from quantum_model import QLSTM
from hybrid_model import Hybrid_QLSTM
from classical_model import LSTM
# define the quantum model

epoch = 14
model_q = QLSTM()
model_q.load_state_dict(torch.load(f"quantum_model/{epoch}.pt"))
model_q.eval()
# define the hybrid model
model_h = Hybrid_QLSTM()
model_h.load_state_dict(torch.load(f"hybrid_model/{epoch}.pt"))
model_h.eval()

# define the classical model
model_c = LSTM()
model_c.load_state_dict(torch.load(f"classical_model/{epoch}.pt"))
model_c.eval()

dataset = data.Dataset(function="sin", interval=.01, n_datapoints=1000, window=4, batch_size=10)
dataset.plot_pred([(model_c, "classical"), (model_h, "hybrid"), (model_q, "quantum")])