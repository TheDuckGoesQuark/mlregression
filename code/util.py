import eli5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from eli5.sklearn import PermutationImportance


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
    x = np.copy(x)
    n_cols = x.shape[1]
    for col_index in range(n_cols):
        col = x[:, col_index]
        factor = np.max(col)
        x[:, col_index] = col / factor

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
            ax.scatter(x_values, y_values, alpha=0.1)
            ax.set_title("X{} against Y{}".format(x_index + 1, y_index + 1))
            ax.set_ylabel("Y{}".format(y_index + 1))
            ax.set_xlabel("X{}".format(x_index + 1))
            current_plot = current_plot + 1

        plt.tight_layout()
        plt.savefig("plots/XsVsY{}.png".format(y_index))
        plt.show()

    # plot ys against each other also
    plt.scatter(y[:, 0], y[:, 1])
    plt.title("Y1 against Y2")
    plt.xlabel("Y1")
    plt.ylabel("Y2")
    plt.savefig("plots/Y1vsY2.png")
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
            rows.append([x_index + 1, y_index + 1, s_rho, s_p])

    fields = ["x", "y", "rho", "p"]
    for item in fields:
        print("{:>5}".format(item), end=" ")

    print("")

    for row in rows:
        print("{:>5}, {:>5}, {:>5.2f}, {:>5.2f}".format(*row))


def split_data(x, y):
    bins = np.linspace(5, 45, 34)
    y_binned = np.digitize(y[:, 0], bins)
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y_binned)


def plot_depth_accuracy(x_train, y_train):
    depths = []
    scores = []
    cross_validator = KFold(n_splits=10, shuffle=True, random_state=42)
    for depth in range(1, 11):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)

        if model.fit(x_train, y_train[:, 0]).tree_.max_depth < depth:
            break

        score = np.mean(cross_val_score(model, x_train, y_train[:, 0], cv=cross_validator, n_jobs=1))
        depths.append(depth)
        scores.append(score)
        print("Depth %i Accuracy: %.5f" % (depth, score))

    coeffecients = np.polyfit(np.log(depths), scores, 2)
    fit = np.poly1d(coeffecients)
    plt.scatter(depths, scores)
    plt.plot(depths, fit(np.log(depths)), "r")
    plt.title("Accuracy at Different Decision Tree Depths")
    plt.xlabel("Decision Tree Depth")
    plt.ylabel("Model Accuracy")
    plt.savefig("plots/depth.png")
    plt.show()


def visualise_error(predicted, actual, variable_name):
    rmse = mean_squared_error(actual, predicted)

    # Distribution
    differences = actual - predicted
    plt.hist(differences, bins=40)
    plt.title("Distribution of prediction differences")
    plt.xlabel("Difference from actual {} value".format(variable_name))
    plt.ylabel("Frequency")
    plt.savefig("plots/{}errordist".format(variable_name))
    plt.show()

    # Scatter
    plt.scatter(predicted, actual)
    plt.title("Predicted {} against real values".format(variable_name))
    plt.xlabel("Predicted {}".format(variable_name))
    plt.ylabel("Actual {}".format(variable_name))
    plt.text(15, 40, "RMSE = %.4f" % rmse, ha='center', va='center')

    # expected line
    coeffecients = np.polyfit(actual, actual, 1)
    fit = np.poly1d(coeffecients)
    plt.plot(actual, fit(actual), "r")

    plt.savefig("plots/{}scatter".format(variable_name))
    plt.show()

    print(rmse)


def visualise_feature_importance(model, title_variable, x_test, y_test):
    print(eli5.format_as_text(
        eli5.explain_weights(PermutationImportance(model, random_state=42).fit(x_test, y_test)))
    )

    importances = model.feature_importances_
    labels = ["X{} - {:4.1f}%".format(i, importances[i - 1] * 100) for i in range(1, 8)]

    patches, texts = plt.pie(importances, wedgeprops=dict(width=0.5), startangle=90, radius=1.2)

    plt.legend(patches, labels, prop={'size': 12}, bbox_to_anchor=(0.74, 0.5), loc="center right", fontsize=8)
    plt.title("Feature Importance for {}".format(title_variable))
    plt.savefig("plots/fi{}.png".format(title_variable))
    plt.show()
