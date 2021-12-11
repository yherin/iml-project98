import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.sparse.construct import rand
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from project_utils import runModel, optimiseModelParams, split_and_scale, Read_data_output_class2_or_testdata, training_with_PCA, split_training_validate, scaling, runModelCV,Read_data_output_class4
#from project_utils import runModel, split_and_scale
from scipy.stats import uniform
np.random.seed(42)

[x, y ] = Read_data_output_class4(filename="npf_train.csv")
#[x,y] = Read_data_output_class2_or_testdata(binary=True, training_data=True, filename = "npf_train.csv")
[x_test,y_test] = Read_data_output_class2_or_testdata(binary=False, training_data=False, filename = "npf_test_hidden.csv")

PCA_num = 30
x = scaling(x)
[x_PCA, pca] = training_with_PCA(x,PCA_num)
[x_train, x_val, y_train, y_val] = train_test_split(x_PCA, y, test_size=0.33)

#Define your model parametrs here and add them to the list

rf = RandomForestClassifier()
rf_params = dict(min_samples_split=[2,3,4,5,6], min_samples_leaf=[1,2,3,4,5,6])

lr = LogisticRegression()
lr_params = dict(C=uniform(loc=0, scale=4), penalty=['l2'], solver=['newton-cg', 'lbfgs'])

nb = GaussianNB()
nb_params = {}

svm = SVC(probability=True)
svm_params = dict(C=uniform(loc=0, scale=4), kernel=['sigmoid', 'rbf', 'poly'])
#xcols = xdf.columns

models = [rf, lr, nb, svm]
model_params = [rf_params, lr_params, nb_params, svm_params]

#Run the models and get the results
#measures = [runModel(m, x_train, x_val, y_train, y_val) for m in models]
#measures = [runModel(m, *split_and_scale(xdf, ydf)) for m in models]
#results = pd.DataFrame(measures, columns=['Name', 'Train Accuracy', 'Train Perplex', 'Valid Accuracy', 'Valid Perplex'])

#model_results: pd.DataFrame = None
#model_dict = {}


#models = [ lr ]
#model_params = [ lr_params ]
model_dict = {}
model_results = None
for m, mp in zip(models, model_params):
    results, md = runModelCV(model=m, model_params=mp, x=x_PCA, y=y, n_iterations=5, k_folds=5)
    if model_results is None:
        model_results = results
    else:
        model_results = pd.concat([model_results, results])
    model_dict.update(md)

#Show results
print("Least perplexed")
print(model_results.sort_values('Validation Perplex', ascending=True).head(5))
print("Most accurate")
print(model_results.sort_values('Validation Accuracy', ascending=False).head(5))
print("Most perplexed")
print(model_results.sort_values('Validation Perplex', ascending=False).head(5))
print("Least accurate")
print(model_results.sort_values('Validation Accuracy', ascending=True).head(5))

pickle.dump(model_results, open('CV_RESULTS_class4.pickle', 'wb'))
