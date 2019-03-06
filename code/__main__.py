import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

import featureselection
import regression

x, y = featureselection.load_inputs_and_outputs("data/ENB2012_data.csv")
# featureselection.visualise(x, y)
# featureselection.spearman(x, y)

# Remove X6 and X8 from features as discussed in report
x = np.delete(x, 5, 1)
x = np.delete(x, 6, 1)

# X3, 4, 5, 7 remain
x_train, x_test, y_train, y_test = regression.split_data(x, y)

x_train = featureselection.normalise(x_train)
y_train = featureselection.normalise(y_train)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)

predictions = tree_reg.predict(x_test)
lin_mse = mean_squared_error(y_test, predictions)
lin_rmse = np.sqrt(lin_mse)

featureselection.plot_scatters(x_test, y_test)
featureselection.plot_scatters(x_test, predictions)

print(lin_rmse)

# file:///home/jordan/Downloads/Hands-On%20Machine%20Learning%20with%20Scikit-Learn%20&%20TensorFlow%20(%20PDFDrive.com%20).pdf
