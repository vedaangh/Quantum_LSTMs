import pennylane as qml
import torch
import torch.nn as nn

# initialise key variables and constants
INPUT_DIM = 1
OUTPUT_DIM = 1
HIDDEN_DIM = 5
vt_dim = INPUT_DIM + HIDDEN_DIM
num_qubits = 4
DEPTH = 2
weight_shapes_dict = {"W": [DEPTH, num_qubits, 3]}
VQCs = ["forget", "input", "update", "output", "hidden", "y_layer"]

VQC_dict = {}


def entangling_layer(n_qubits, latency):
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + latency) % n_qubits])


for vqc in VQCs:
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device=dev, interface="pytorch", diff_method="backprop")
    def circuit(inputs, W):
        for i in range(num_qubits):
            qml.RY(qml.math.arctan(inputs[i]), wires=i)
            qml.RZ(qml.math.arctan(inputs[i] ** 2), wires=i)
        for d in range(DEPTH):
            entangling_layer(num_qubits, 1)
            entangling_layer(num_qubits, 2)
            for qbit in range(num_qubits):
                qml.Rot(W[d, qbit, 0], W[d, qbit, 1], W[d, qbit, 2], wires=qbit)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    VQC_dict[vqc] = circuit


weights = torch.Tensor([[[0, 0, 0] for _ in range(4)] for idx in range(DEPTH)])
print(qml.draw(VQC_dict["forget"])([0, 0, 0, 0], weights))



class Hybrid_QLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.VQC_layers = nn.ModuleList()
        for circ in VQCs:
            self.VQC_layers.append(qml.qnn.TorchLayer(VQC_dict[circ], weight_shapes_dict))

        self.classical_layers_in = nn.ModuleList()
        for _ in range(6):
            scaler_in = nn.Linear(vt_dim, num_qubits)
            self.classical_layers_in.append(scaler_in)


        self.classical_layers_out = nn.ModuleList()
        for _ in range(4):
            scaler_out = nn.Linear(num_qubits, vt_dim)
            self.classical_layers_out.append(scaler_out)


        self.classical_layers_out.append(nn.Linear(num_qubits, HIDDEN_DIM))
        self.classical_layers_out.append(nn.Linear(num_qubits, OUTPUT_DIM))


    def forward(self, input_seq):
        y_seq = []
        batch_size, seq_length, feature_size = input_seq.size()  # shape is  seq x features assume no batches

        ct = torch.zeros(batch_size, vt_dim)  # .to(device)
        ht = torch.zeros(batch_size, HIDDEN_DIM)  # .to(device)

        for t in range(seq_length):
            x = input_seq[:, t, :]  # take all features input at timestep t

            vt = torch.concat((x, ht), dim=1)

            ft = torch.sigmoid(self.classical_layers_out[0](self.VQC_layers[0](self.classical_layers_in[0](vt))))
            # forget
            it = torch.sigmoid(self.classical_layers_out[1](self.VQC_layers[1](self.classical_layers_in[1](vt))))
            # input
            gt = torch.tanh(self.classical_layers_out[2](self.VQC_layers[2](self.classical_layers_in[2](vt))))
            # update
            ot = torch.sigmoid(self.classical_layers_out[3]((self.VQC_layers[3](self.classical_layers_in[3](vt)))))
            # output

            ct = ft * ct + (it * gt)
            ht = self.classical_layers_out[4](self.VQC_layers[4](self.classical_layers_in[4](ot * torch.tanh(ct))))
            yt = self.classical_layers_out[5](self.VQC_layers[5](self.classical_layers_in[5](ot * torch.tanh(ct))))
            y_seq.append(yt)

        y = y_seq[-1]  # we are predicting T_n+1 | T_1->n

        return y


print(f"number of parameters in hybrid model: {sum(p.numel() for p in Hybrid_QLSTM().parameters() if p.requires_grad)}")