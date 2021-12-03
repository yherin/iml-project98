import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns


def CV_evaluate_model(model,X,Y,k):

    scores = cross_val_score(model, X,Y, cv=k)
    average_score = np.average(scores, axis=0)
    return average_score

def plot_boundaries(X1,X2, Y,regr_SVM):
    h = .02  # step size in the mesh
    x_min, x_max = X1.min() - 1, X1.max() + 1
    y_min, y_max = X2.min() - 1,X2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = regr_SVM.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X1,X2, c = Y, cmap=plt.cm.coolwarm)
    plt.show()


npf = pd.read_csv("npf_train.csv")
npf = npf.set_index("date")
## Tell that "class4" is categorical variable. (R does this automatically.)
npf["class4"] = npf["class4"].astype("category")
## Here date column was converted to index and we do not need to get rid of it.
npf = npf.drop("id",axis=1)
#npf.describe()
npf = npf.drop("partlybad",axis=1)

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
explained_var = y_training


scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
pca = PCA()
x_new = pca.fit_transform(X)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
##This will open on browser, is more interactive
# fig = px.area(
#     x=range(1, exp_var_cumul.shape[0] + 1),
#     y=exp_var_cumul,
#     labels={"x": "# Components", "y": "Explained Variance"}
# )
# fig.show()
###

components=range(1, exp_var_cumul.shape[0] + 1)
explained_var=exp_var_cumul
plt.plot(components, explained_var)
plt.show()
plt.plot(components[0:35], explained_var[0:35])
plt.show()

num_PCA = 35
PCA_x_training = x_new[:, 0:num_PCA ]

########################### Models #################

log_reg = LogisticRegression()
regr_SVM = svm.SVC(kernel = 'rbf')
regr_tree = DecisionTreeRegressor(max_depth=10)
regr_forest = RandomForestRegressor(max_depth=10, random_state=0)


models = np.hstack((log_reg, regr_SVM,regr_tree,regr_forest))
CV_test = np.empty([len(models)])
for i in range(len(models)):
    CV_test[i] = CV_evaluate_model(models[i],PCA_x_training,y_training,10)
print(CV_test)

##########################SVM
regr_SVM = svm.SVC(kernel = 'rbf')
regr_SVM.fit(PCA_x_training, y_training)
y_training_pred_SVM = regr_SVM.predict(PCA_x_training)

### this only works with num_PCA = 2
#plot_boundaries(PCA_x_training[:, 0],PCA_x_training[:, 1], y_training,regr_SVM)


N=5
for i in range(N):
    for j in range(N):
        if i==j:
                continue
        else:
            plt.subplot(N, N, j+1 +N*i)
            plt.scatter(x_new[:, i],x_new[:, j], c = y_training, cmap=plt.cm.coolwarm, s=1)
plt.show()
