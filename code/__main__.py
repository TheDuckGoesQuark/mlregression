import featureselection
import numpy as np

x, y = featureselection.load_inputs_and_outputs("data/ENB2012_data.csv")
featureselection.visualise(x, y)
featureselection.spearman(x, y)

# Add column of ones
x = np.c_[np.ones_like(x), x]
