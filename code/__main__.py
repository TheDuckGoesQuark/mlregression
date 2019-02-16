import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_inputs_and_outputs(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    inputs = data[:, 0:8]
    outputs = data[:, 8:]
    return inputs, outputs


x, y = load_inputs_and_outputs("data/ENB2012_data.csv")
x = np.c_[np.ones_like(x), x]
