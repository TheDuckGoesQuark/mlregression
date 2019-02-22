import featureselection
import numpy as np

x, y = featureselection.load_inputs_and_outputs("data/ENB2012_data.csv")
#featureselection.visualise(x, y)
# featureselection.spearman(x, y)

# Remove X6 and X8 from features as discussed in report
x = np.delete(x, 5, 1)
x = np.delete(x, 6, 1)

featureselection.recursive_feature_elimination(x, y)

