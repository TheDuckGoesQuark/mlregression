import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from sklearn.feature_selection import RFE
from sklearn.metrics import mutual_info_score
from sklearn.svm import LinearSVC, LinearSVR


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

    plot_rows = n_cols // 2
    plot_rows += n_cols % 2
    plot_cols = 2

    position = range(1, n_cols + 1)
    fig = plt.figure()

    for col_index in range(n_cols):
        col_values = x[:, col_index]
        ax = fig.add_subplot(plot_rows, plot_cols, position[col_index])
        ax.hist(col_values)
        ax.set_title("Distribution of variable {}{}".format(variable_name, col_index + 1))
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Value")

    plt.tight_layout()
    plt.savefig("plots/{}Dist.png".format(variable_name))
    plt.show()


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


def plot_scatters(x, y):
    n_x_cols = x.shape[1]
    n_y_cols = y.shape[1]

    for y_index in range(n_y_cols):
        num_plots = n_x_cols
        plot_rows = num_plots // 3
        plot_rows += num_plots % 3
        plot_cols = 3

        position = range(1, num_plots + 1)
        fig = plt.figure(figsize=(10, 8))

        current_plot = 0
        for x_index in range(n_x_cols):
            x_values = x[:, x_index]
            y_values = y[:, y_index]

            ax = fig.add_subplot(plot_rows, plot_cols, position[current_plot])
            ax.scatter(x_values, y_values)
            ax.set_title("X{} against Y{}".format(x_index + 1, y_index + 1))
            ax.set_ylabel("Y{}".format(y_index + 1))
            ax.set_xlabel("X{}".format(x_index + 1))
            current_plot = current_plot + 1

        plt.tight_layout()
        plt.savefig("plots/XsVsY{}.png".format(y_index))
        plt.show()


def visualise(x, y):
    plot_distributions(x, "X")
    plot_distributions(y, "Y")

    norm_x = normalise(x)
    norm_y = normalise(y)
    plot_scatters(norm_x, norm_y)


def spearman(x, y):
    n_x_cols = x.shape[1]
    n_y_cols = y.shape[1]

    rows = []
    for x_index in range(n_x_cols):
        for y_index in range(n_y_cols):
            x_col = x[:, x_index]
            y_col = y[:, y_index]
            s_rho, s_p = stats.spearmanr(x_col, y_col)
            rows.append([x_index+1, y_index+1, s_rho, s_p])

    fields = ["x", "y", "rho", "p"]
    for item in fields:
        print("{:>5}".format(item), end=" ")

    print("")

    for row in rows:
        print("{:>5}, {:>5}, {:>5.2f}, {:>5.2f}".format(*row))


def recursive_feature_elimination(x, y):
    svm = LinearSVR()

    for y_index in range(y.shape[1]):
        y_col = y[:, y_index]
        rfe = RFE(svm, 4)
        rfe = rfe.fit(x, y_col)
        print(rfe.support_)
        print(rfe.ranking_)
