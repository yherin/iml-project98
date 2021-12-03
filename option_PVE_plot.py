

import numpy as np
import pandas as pd
from pca import pca


npf = pd.read_csv("npf_train.csv")
npf = npf.set_index("date")
## Tell that "class4" is categorical variable. (R does this automatically.)
npf["class4"] = npf["class4"].astype("category")
## Here date column was converted to index and we do not need to get rid of it.
npf = npf.drop("id",axis=1)
#npf.describe()
npf = npf.drop("partlybad",axis=1)
## If you don't use dtype="object" array will cut strings...
class2 = np.array(["event"]*npf.shape[0],dtype="object")
class2[npf["class4"]=="nonevent"] = "nonevent"
npf["class2"] = class2
npf["class2"] = npf["class2"].astype("category")
y_training = npf["class2"]
y_training_d = pd.get_dummies(y_training, prefix=['event', "nonevent"])
y_training = y_training_d.iloc[:,0]
x_training = npf.drop("class2",axis=1)
x_training = x_training.drop("class4",axis=1)
####test data ###############

npf1 = pd.read_csv("npf_test_hidden.csv")
npf1 = npf1.drop("date",axis=1)
npf1 = npf1.drop("id",axis=1)
npf1 = npf1.drop("class4",axis=1)
npf1 = npf1.drop("partlybad",axis=1)
x_testing = npf1

X = x_training
y = y_training

# Initialize to reduce the data up to the number of componentes that explains 95% of the variance.
model = pca(n_components=0.95)

# Or reduce the data towards 2 PCs
model = pca(n_components=8)

# Fit transform
results = model.fit_transform(X)

# Plot explained variance
fig, ax = model.plot()

# Scatter first 2 PCs
fig, ax = model.scatter()

# Make biplot with the number of features
fig, ax = model.biplot(n_feat=4,legend=True)

