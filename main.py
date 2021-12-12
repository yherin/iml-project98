import pandas as pd
import numpy as np
import pickle
import time
import os
from matplotlib import pyplot as plt
from scipy.sparse.construct import rand
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from dummy_classifier import DummyClassifier
from project_utils import runModel, optimiseModelParams, split_and_scale, Read_data_output_class2_or_testdata, \
    training_with_PCA, split_training_validate, scaling, runModelCV, Read_data_output_class4, runModelStepwiseSelection
#from project_utils import runModel, split_and_scale
from scipy.stats import uniform
np.random.seed(42)


#[x,y] = Read_data_output_class2_or_testdata(binary=True, training_data=True, filename = "npf_train.csv")
#[x_test,y_test] = Read_data_output_class2_or_testdata(binary=False, training_data=False, filename = "npf_test_hidden.csv")
x,y, enc = Read_data_output_class4("npf_train.csv")

#save the encoded training data incase we would need it later
pickle.dump(enc, open('y_encoder.pickle', 'wb'))



question = 'Feature selection: PCA (p) or Stepwise (s) ? [p/s]:\t'
choice = input(question)
if choice == 'p': #pca

    PCA_num = 15
    x = scaling(x)
    [x_PCA, pca] = training_with_PCA(x,PCA_num)
    [x_train, x_val, y_train, y_val] = train_test_split(x_PCA, y, test_size=0.33)

    #save encoded training data for later use
    pickle.dump(pca, open('pca_model.pickle', 'wb'))
    pickle.dump(x_PCA, open('X_train_PCA.pickle', 'wb'))
    pickle.dump(y, open('y_train.pickle', 'wb'))

    adjusted_weights = {0: 0.24, 1: 0.05, 2: 0.16, 3: 0.55}

    c = np.linspace(0,2,100)
    l1 = np.linspace(0,1,100)
    rf = RandomForestClassifier()
    #rf_params = dict(min_samples_split=[2,3,4,5,6], min_samples_leaf=[1,2,3,4,5,6], class_weight=[adjusted_weights])
    rf_params = dict(min_samples_split=[2,3,4,5,6], min_samples_leaf=[2,3,4,5,6], class_weight=[adjusted_weights])

    lr = LogisticRegression(max_iter=1000)
    lr_params = dict( penalty=['l2'], solver=['lbfgs', 'newton-cg', 'sag'], class_weight=[adjusted_weights], multi_class=['multinomial'])

    lre = LogisticRegression(max_iter=1000)
    lre_params = dict( l1_ratio=l1, penalty=['elasticnet', 'l1', 'l2'], solver=['saga'], class_weight=[adjusted_weights], multi_class=['multinomial'])

    nb = GaussianNB(priors=np.array([0.24, 0.05, 0.16, 0.55]))
    nb_params = dict()

    svm = SVC()
    svm_params = dict(kernel=['sigmoid', 'rbf', 'poly'], degree=[2,3], class_weight=[adjusted_weights], probability=[True], decision_function_shape=['ovo', 'ovr'])
    #svm_params = dict(C=c,  kernel=['sigmoid', 'rbf', 'poly'], class_weight=[adjusted_weights], probability=[True])


    models = [rf,  lre,  nb, svm]
    model_params = [rf_params, lre_params, nb_params, svm_params]


elif choice == 's': #stepwise
    """ IMPLEMENT STEPWISE SELECTION HERE"""
    #stepwise_results = runModelStepwiseSelection(x=x, y=y, )
    #is this stepwise going to be computed for every model?
    rf_ = RandomForestClassifier(n_estimators=50, random_state=42, class_weight={1: 0.55, 0:0.45}, min_samples_leaf=5, min_samples_split=5)
    lr_ = LogisticRegression(max_iter=10)
    nb_ = GaussianNB()
    svm_ = SVC(probability=True)
    models_ = [rf_, lr_, nb_, svm_]
    model_params = [dict(), dict(), dict(), dict()] #need this or it might crash
    stepWiseSelection = []
    for m in models_:
        stepWiseSelection.append(runModelStepwiseSelection(m, x, y))    
else:
    raise ValueError('Invalid input: choose p or s')



#Define your model parametrs here and add them to the list

#class weights in training set
#II: 117, Ia: 29, Ib: 83, nonevent: 229
#this is artificially 50/50, we know in the training data there are slighty more nonevents
#II = 0 - weight 0.255
#Ia = 1 - weight 0.063
#Ib = 2 - weight 0.181
#nonevent = 3 - weight 0.5




model_dict = {}
model_results = None
i = 0
for m, mp in zip(models, model_params):
    if (choice == 'p'):
        results, md = runModelCV(model=m, model_params=mp, x=x_PCA, y=y, n_iterations=5, k_folds=5)
    else:
        x_loc = scaling(x.iloc[:,np.array(stepWiseSelection[i]['features'])])
        results, md = runModelCV(model=m, model_params=mp, x=x_loc, y=y, n_iterations=5, k_folds=5)
        i += 1
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
#save the actual models and the performance results here

ts = time.strftime('%d%m%y_%H%M%S')
unam = os.getlogin()
pickle.dump(model_dict, open(f'models-{ts}-{unam}.pickle', 'wb'))
pickle.dump(model_results, open(f'CV_RESULTS-{ts}-{unam}.pickle', 'wb'))
