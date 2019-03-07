import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import featureselection
import regression

x, y = featureselection.load_inputs_and_outputs("data/ENB2012_data.csv")

x_train, x_test, y_train, y_test = regression.split_data(x, y)

# Feature selection and visualisation

featureselection.visualise(x_train, y_train)
featureselection.spearman(x_train, y_train)

# Remove X6 and X8 from features as discussed in report
x_train = np.delete(x, 5, 1)
x_train = np.delete(x, 6, 1)

# X1, 2, 3, 4, 5, 7 remain

x_train = featureselection.normalise(x_train)
y_train = featureselection.normalise(y_train)

