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

device = torch.device("cuda")
# key constants
lr = 1e-3
# define the quantum model
model_q = QLSTM()
# model_q.load_state_dict(torch.load("quantum_model/1.pt"))
optim_q = torch.optim.RMSprop(model_q.parameters(), lr=lr)

# define the hybrid model
model_h = Hybrid_QLSTM()
# model_h.load_state_dict(torch.load("hybrid_model/1.pt"))
optim_h = torch.optim.RMSprop(model_h.parameters(), lr=lr)

# define the classical model
model_c = LSTM()
# model_c.load_state_dict(torch.load("classical_model/1.pt"))
optim_c = torch.optim.RMSprop(model_c.parameters(), lr=lr)

# initialising key features
n_epochs = 100
criterion = nn.MSELosssh()
dataset = data.Dataset(function="sin", interval=.01, n_datapoints=1000, window=4, batch_size=10)

training_data = dataset.train_dataset

for epoch in range(n_epochs):

    # shuffle training data to reduce bias
    random.shuffle(training_data)

    # save models
    torch.save(model_q.state_dict(), f"quantum_model/{epoch}.pt")
    torch.save(model_h.state_dict(), f"hybrid_model/{epoch}.pt")
    torch.save(model_c.state_dict(), f"classical_model/{epoch}.pt")

    for datapoint in training_data:
        X, y = datapoint[0], datapoint[1]

        # train quantum model
        y_q = model_q(X)
        loss_q = criterion(y_q, y)
        print(f"quantum model loss:{loss_q}")
        loss_q.backward()
        optim_q.step()

        # train hybrid model
        y_h = model_h(X)
        loss_h = criterion(y_h, y)
        print(f"hybrid model loss:{loss_h}")
        loss_h.backward()
        optim_h.step()

        # train classical model
        y_c = model_c(X)
        loss_c = criterion(y_c, y)
        print(f"classical model loss:{loss_c}")

        loss_c.backward()
        optim_c.step()
    """
    y_plot = [model(inp).detach() for inp in input_arr]
    x_plot = [_ for _ in range(len(y_plot))]
    plt.plot(x_plot, y_plot)
    plt.show()
    """


