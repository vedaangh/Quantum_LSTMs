import torch
import matplotlib.pyplot as plt
import random
import math
from scipy import special

random.seed(10)


def bessel_order_2(x): return special.yn(1, x)


class Dataset:

    def __init__(self, function, interval, n_datapoints, window,
                 shuffle=True, train_split=2 / 3, plot=True, batch_size=1):

        # locate function
        functions = {"sin": math.sin, "bessel_2": bessel_order_2}
        function = functions[function]

        self.dataset = [(torch.Tensor([[function(i * interval)] for i in range(x, x + window)]),
                         torch.Tensor([function((x + window) * interval)])) for x in range(n_datapoints)]

        self.init_dataset = [(torch.Tensor([[function(i * interval)] for i in range(x, x + window)]),
                              torch.Tensor([function((x + window) * interval)])) for x in range(n_datapoints)]


        if plot:
            x_plot = [_ * interval for _ in range(n_datapoints)]
            y_plot = [function(x) for x in x_plot]
            plt.plot(x_plot, y_plot)
            plt.show()

        if shuffle:
            random.shuffle(self.dataset)

        train_dataset = self.dataset[:int(train_split * n_datapoints)]
        self.train_dataset = []
        for idx in range(0, len(train_dataset)-batch_size+1, batch_size):
            x = []
            y = []
            for batch_idx in range(batch_size):
                x.append(train_dataset[idx + batch_idx][0])
                y.append(train_dataset[idx + batch_idx][1])
            self.train_dataset.append((torch.stack(x), torch.stack(y)))

        self.test_dataset = self.dataset[int(train_split * n_datapoints):]


    def plot_pred(self, models):
        results = []
        labels = []

        for model_tup in models:
            model = model_tup[0]
            labels.append(model_tup[1])
            y_plot = []
            for point in self.init_dataset:
                x = point[0]
                x = x.reshape(1, x.size()[0], x.size()[1])
                y_plot.append(model(x).detach())
            results.append(y_plot)
        y_truth = []
        for point in self.init_dataset:
            y_truth.append(point[1])
        x_plot = [i for i in range(len(results[0]))]
        for y_idx in range(len(results)):
            plt.plot(x_plot, results[y_idx], label=labels[y_idx])
        plt.plot(x_plot, y_truth,"--", label="ground truth")
        plt.legend()
        plt.show()

