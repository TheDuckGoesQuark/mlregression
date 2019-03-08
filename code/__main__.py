import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import util

# Loading and Cleaning

x, y = util.load_inputs_and_outputs("data/ENB2012_data.csv")

x_train, x_test, y_train, y_test = util.split_data(x, y)

# Feature selection and Visualisation

util.visualise(x_train, y_train)
util.spearman(x_train, y_train)

# Remove X8 from features as discussed in report
x_train = np.delete(x_train, 7, 1)
x_test = np.delete(x_test, 7, 1)

# Training and optimisation
util.plot_depth_accuracy(x_train, y_train)

# Evaluation
y1_model = DecisionTreeRegressor(max_depth=6, random_state=42)
y1_model.fit(x_train, y_train[:, 0])

y2_model = DecisionTreeRegressor(max_depth=6, random_state=42)
y2_model.fit(x_train, y_train[:, 1])

util.visualise_feature_importance(y1_model, "Y1")
util.visualise_feature_importance(y2_model, "Y2")

y1_predicted = y1_model.predict(x_test)
y2_predicted = y2_model.predict(x_test)

util.visualise_error(y1_predicted, y_test[:, 0], "Y1")
util.visualise_error(y2_predicted, y_test[:, 1], "Y2")


