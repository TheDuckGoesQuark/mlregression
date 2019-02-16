import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_inputs_and_outputs(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=True, invalid_raise=True)
    inputs = data[:, 0:8]
    outputs = data[:, 8:]
    return inputs, outputs


def plot_distributions(x):
    n_cols = x.shape[1]
    for col_index in range(n_cols):
        fig = plt.figure(col_index)
        col_values = x[:, col_index]
        plt.hist(col_values)
        plt.title("Distribution of variable X{}".format(col_index + 1))
        plt.ylabel("Frequency")
        plt.xlabel("Value")
        plt.show()


x, y = load_inputs_and_outputs("data/ENB2012_data.csv")
plot_distributions(x)

# Add column of ones
x = np.c_[np.ones_like(x), x]
