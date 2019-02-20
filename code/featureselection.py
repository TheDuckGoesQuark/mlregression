import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats


def load_inputs_and_outputs(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=True, invalid_raise=True)
    inputs = data[:, 0:8]
    outputs = data[:, 8:]
    return inputs, outputs


def plot_distributions(x, variable_name):
    """
    Plots the distribution of all the columns in X in a series of subplots
    :param x: matrix where each column represents the set of values for a variable
    :param variable_name: X/Y to print as title
    """
    n_cols = x.shape[1]
    print(n_cols)

    plot_rows = n_cols // 2
    plot_rows += n_cols % 2
    plot_cols = 2

    position = range(1, n_cols+1)
    fig = plt.figure()

    for col_index in range(n_cols):
        col_values = x[:, col_index]
        ax = fig.add_subplot(plot_rows, plot_cols, position[col_index])
        ax.hist(col_values)
        ax.set_title("Distribution of variable {}{}".format(variable_name, col_index + 1))
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Value")

    plt.tight_layout()


def normalise(x):
    """
    Normalise each column in x by diving all values in a column by the maximum value of that column
    :param x: a 2D matrix of values
    :return: The normalised matrix
    """
    n_cols = x.shape[1]
    for col_index in range(n_cols):
        col = x[:, col_index]
        factor = np.max(col)
        x[:, col_index] = col / factor

    return x


def mean_normalise(x):
    """
    Normalise each column in x by subtracting the mean from each value and diving by the range.
    :param x: a 2D matrix of values
    :return: The normalised matrix
    """
    n_cols = x.shape[1]
    for col_index in range(n_cols):
        col = x[:, col_index]
        mean = np.mean(col)
        range_of_col = np.max(col) - np.min(col)
        x[:, col_index] = (col - mean) / range_of_col

    return x


def plot_scatter(x, y, figure_n, x_n, y_n):
    plt.figure(figure_n)
    plt.scatter(x, y)
    plt.title("X{} against Y{}".format(x_n, y_n))
    plt.ylabel("Y{}".format(y_n))
    plt.xlabel("X{}".format(x_n))
    plt.show()


def plot_scatters(x, y):
    n_x_cols = x.shape[1]
    n_y_cols = y.shape[1]

    fig_n = 0
    for y_index in range(n_y_cols):
        for x_index in range(n_x_cols):
            plot_scatter(x[:, x_index], y[:, y_index], fig_n, x_index + 1, y_index + 1)
            fig_n = fig_n + 1


def visualise(x, y):
    plot_distributions(x, "X")
    plot_distributions(y, "Y")

    norm_x = normalise(x)
    norm_y = normalise(y)
    plot_scatters(norm_x, norm_y)


def spearman(x, y):
    n_x_cols = x.shape[1]
    n_y_cols = y.shape[1]

    for y_index in range(n_y_cols):
        for x_index in range(n_x_cols):
            x_col = x[:, x_index]
            y_col = y[:, y_index]
            s_rho, s_p = stats.spearmanr(x_col, y_col)
            p_rho, p_p = stats.pearsonr(x_col, y_col)
            print("Spearman - X{} : Y{} has Rho {} and p {}".format(x_index + 1, y_index + 1, s_rho, s_p))
            print("Pearson  - X{} : Y{} has Rho {} and p {}".format(x_index + 1, y_index + 1, p_rho, p_p))
