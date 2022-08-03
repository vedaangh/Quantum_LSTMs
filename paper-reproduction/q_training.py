import math
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from torch.utils import tensorboard
import torch.nn as nn
from model import QLSTM
import random

device = torch.device("cuda")

model = QLSTM()
model.load_state_dict(torch.load("state_dict_model_52.pt"))
optim = torch.optim.RMSprop(model.parameters())

n_epochs = 100


criterion = nn.MSELoss()
sin_dataset = [(torch.Tensor([[math.sin(i/200)] for i in range(x, x+4)]), torch.Tensor([math.sin((x+4)/200)])) for x in range(2400)]
random.shuffle(sin_dataset)
sin_train = sin_dataset[:2*len(sin_dataset)//3]  # first 2 thirds
sin_test = sin_dataset[2*len(sin_dataset)//3:]  # last third

plt.plot([_/200 for _ in range(len(sin_dataset))], [sin_dataset[n][0][0] for n in range(len(sin_dataset))])
plt.show()
batch_size = 10

for epoch in range(n_epochs):
    print(list(model.parameters( )))
    print(epoch)
    random.shuffle(sin_train)

    torch.save(model.state_dict(), f"model_{epoch+52}.pt")
    for idx in range(4, len(sin_train)-batch_size, batch_size):  # looking back at last 4 values
        input_seq = []
        y_truth = []
        for batch_idx in range(batch_size):
            input_seq.append(sin_train[idx+batch_idx][0])
            y_truth.append(sin_train[idx+batch_idx][1])
        input_seq = torch.stack(input_seq)
        y_truth = torch.stack(y_truth)


        y_pred = model(input_seq)
        print(f"y_truth: {y_truth}, y_pred:{y_pred}")
        loss = criterion(y_pred, y_truth)

        print(f"loss:{loss}")
        loss.backward()
        optim.step()

    input_arr = []
    for idx in range(4, len(sin_test), 1):  # looking back at last 4 values
        input_seq = []
        for batch_idx in range(1):
            input_seq.append(sin_test[idx + batch_idx][0])
        input_seq = torch.stack(input_seq)
        input_arr.append(input_seq)
    print("onono")
    """
    y_plot = [model(inp).detach() for inp in input_arr]
    x_plot = [_ for _ in range(len(y_plot))]
    plt.plot(x_plot, y_plot)
    plt.show()
    """

    torch.save(model.state_dict(), f"state_dict_model_{epoch}.pt")

    print(model.parameters())

