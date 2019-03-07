import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import util

# Loading and Cleaning

x, y = util.load_inputs_and_outputs("data/ENB2012_data.csv")

x_train, x_test, y_train, y_test = util.split_data(x, y)

# Feature selection and Visualisation

# util.visualise(x_train, y_train)
# util.spearman(x_train, y_train)

# Remove X8 from features as discussed in report
x_train = np.delete(x_train, 7, 1)
x_test = np.delete(x_test, 7, 1)

# Training
model = DecisionTreeRegressor()
model.fit(x_train, y_train[:, 0])


predicted = model.predict(x_test)

df = pd.DataFrame({'Actual': y_test[:, 0], 'Predicted': predicted})
print(df)
