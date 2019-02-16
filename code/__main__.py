import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_inputs_and_outputs(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=True, invalid_raise=True)
    inputs = data[:, 0:8]
    outputs = data[:, 8:]
    return inputs, outputs


def plot_distributions(x, variable_name):
    n_cols = x.shape[1]
    for col_index in range(n_cols):
        plt.figure(col_index)
        col_values = x[:, col_index]
        plt.hist(col_values)
        plt.title("Distribution of variable {}{}".format(variable_name, col_index + 1))
        plt.ylabel("Frequency")
        plt.xlabel("Value")
        plt.show()


def normalise(x):
    factor = np.max(x)
    return x / factor


def plot_scatter(x, y, figure_n):
    # TODO add labels
    plt.figure(figure_n)
    plt.scatter(x, y)
    plt.show()


def plot_scatters(x, y):
    n_x_cols = x.shape[1]
    n_y_cols = y.shape[1]

    fig_n = 0
    for y_index in range(n_y_cols):
        for x_index in range(n_x_cols):
            plot_scatter(x[:, x_index], y[:, y_index], fig_n)
            fig_n = fig_n + 1


x, y = load_inputs_and_outputs("data/ENB2012_data.csv")
plot_distributions(x, "X")
plot_distributions(y, "Y")

norm_x = normalise(x)
norm_y = normalise(y)

plot_scatters(norm_x, norm_y)

# Add column of ones
x = np.c_[np.ones_like(x), x]
