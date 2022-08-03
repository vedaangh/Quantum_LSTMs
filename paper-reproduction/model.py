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
NUM_QUBITS = INPUT_DIM+HIDDEN_DIM


dev1 = qml.device("default.qubit", wires=NUM_QUBITS)
dev2= qml.device("default.qubit", wires=NUM_QUBITS)
dev3 = qml.device("default.qubit", wires=NUM_QUBITS)
dev4 = qml.device("default.qubit", wires=NUM_QUBITS)

dev5 = qml.device("default.qubit", wires=NUM_QUBITS)
dev6 = qml.device("default.qubit", wires=NUM_QUBITS)

DEPTH = 2


def entangling_layer(n_qubits, latency):
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + latency) % n_qubits])


@qml.qnode(device=dev1, interface="pytorch", diff_method="backprop")
def vt_to_ht(inputs, W):
    for i in range(NUM_QUBITS):
        qml.RY(qml.math.arctan(inputs[i]), wires=i)
        qml.RZ(qml.math.arctan(inputs[i] ** 2), wires=i)


    for d in range(DEPTH):
        entangling_layer(NUM_QUBITS, 1)
        entangling_layer(NUM_QUBITS, 2)
        for qbit in range(NUM_QUBITS):
            qml.Rot(W[d, qbit, 0], W[d, qbit, 1], W[d, qbit, 2], wires=qbit)
    return [qml.expval(qml.PauliZ(i)) for i in range(HIDDEN_DIM)]

@qml.qnode(device=dev2, interface="pytorch", diff_method="backprop")
def vt_to_yt_forget(inputs, W):
    for i in range(NUM_QUBITS):
        qml.RY(qml.math.arctan(inputs[i]), wires=i)
        qml.RZ(qml.math.arctan(inputs[i] ** 2), wires=i)


    for d in range(DEPTH):
        entangling_layer(NUM_QUBITS, 1)
        entangling_layer(NUM_QUBITS, 2)
        for qbit in range(NUM_QUBITS):
            qml.Rot(W[d, qbit, 0], W[d, qbit, 1], W[d, qbit, 2], wires=qbit)
    return [qml.expval(qml.PauliZ(i)) for i in range(OUTPUT_DIM)]

@qml.qnode(device=dev3, interface="pytorch", diff_method="backprop")
def vt_to_yt_input(inputs, W):
    for i in range(NUM_QUBITS):
        qml.RY(qml.math.arctan(inputs[i]), wires=i)
        qml.RZ(qml.math.arctan(inputs[i] ** 2), wires=i)


    for d in range(DEPTH):
        entangling_layer(NUM_QUBITS, 1)
        entangling_layer(NUM_QUBITS, 2)
        for qbit in range(NUM_QUBITS):
            qml.Rot(W[d, qbit, 0], W[d, qbit, 1], W[d, qbit, 2], wires=qbit)
    return [qml.expval(qml.PauliZ(i)) for i in range(OUTPUT_DIM)]

@qml.qnode(device=dev4, interface="pytorch", diff_method="backprop")
def vt_to_yt_output(inputs, W):
    for i in range(NUM_QUBITS):
        qml.RY(qml.math.arctan(inputs[i]), wires=i)
        qml.RZ(qml.math.arctan(inputs[i] ** 2), wires=i)


    for d in range(DEPTH):
        entangling_layer(NUM_QUBITS, 1)
        entangling_layer(NUM_QUBITS, 2)
        for qbit in range(NUM_QUBITS):
            qml.Rot(W[d, qbit, 0], W[d, qbit, 1], W[d, qbit, 2], wires=qbit)
    return [qml.expval(qml.PauliZ(i)) for i in range(OUTPUT_DIM)]

@qml.qnode(device=dev5, interface="pytorch", diff_method="backprop")
def vt_to_yt_update(inputs, W):
    for i in range(NUM_QUBITS):
        qml.RY(qml.math.arctan(inputs[i]), wires=i)
        qml.RZ(qml.math.arctan(inputs[i] ** 2), wires=i)


    for d in range(DEPTH):
        entangling_layer(NUM_QUBITS, 1)
        entangling_layer(NUM_QUBITS, 2)
        for qbit in range(NUM_QUBITS):
            qml.Rot(W[d, qbit, 0], W[d, qbit, 1], W[d, qbit, 2], wires=qbit)
    return [qml.expval(qml.PauliZ(i)) for i in range(OUTPUT_DIM)]

@qml.qnode(device=dev6, interface="pytorch", diff_method="backprop")
def vt_to_yt(inputs, W):
    for i in range(NUM_QUBITS):
        qml.RY(qml.math.arctan(inputs[i]), wires=i)
        qml.RZ(qml.math.arctan(inputs[i] ** 2), wires=i)


    for d in range(DEPTH):
        entangling_layer(NUM_QUBITS, 1)
        entangling_layer(NUM_QUBITS, 2)
        for qbit in range(NUM_QUBITS):
            qml.Rot(W[d, qbit, 0], W[d, qbit, 1], W[d, qbit, 2], wires=qbit)
    return [qml.expval(qml.PauliZ(i)) for i in range(OUTPUT_DIM)]


weights = torch.Tensor([[[0, 0, 0] for _ in range(4)] for idx in range(DEPTH)])

print(qml.draw(vt_to_yt)([0, 0, 0, 0], weights))
weight_shapes_dict = {"W": [DEPTH, NUM_QUBITS, 3]}



class QLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.forget_l = qml.qnn.TorchLayer(vt_to_yt_forget, weight_shapes_dict)
        self.input_l = qml.qnn.TorchLayer(vt_to_yt_input, weight_shapes_dict)
        self.update_l = qml.qnn.TorchLayer(vt_to_yt_update, weight_shapes_dict)
        self.output_l = qml.qnn.TorchLayer(vt_to_yt_output, weight_shapes_dict)
        self.y_l = qml.qnn.TorchLayer(vt_to_yt, weight_shapes_dict)
        self.ht_l = qml.qnn.TorchLayer(vt_to_ht, weight_shapes_dict)
    def forward(self, input_seq):
        y_seq = []
        batch_size, seq_length, feature_size = input_seq.size()  # shape is  seq x features assume no batches

        ct = torch.zeros(batch_size, NUM_QUBITS)  # .to(device)
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