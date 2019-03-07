import numpy as np

import featureselection
import regression

# Loading and Cleaning

x, y = featureselection.load_inputs_and_outputs("data/ENB2012_data.csv")

x_train, x_test, y_train, y_test = regression.split_data(x, y)

# Feature selection and Visualisation

featureselection.visualise(x_train, y_train)
featureselection.spearman(x_train, y_train)

# Remove X8 from features as discussed in report
x_train = np.delete(x, 7, 1)

x_train = featureselection.normalise(x_train)
y_train = featureselection.normalise(y_train)

# Selection and Training





