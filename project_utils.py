from __future__ import annotations
from typing import Tuple
from scipy.sparse.construct import rand
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from warnings import warn
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from dummy_classifier import DummyClassifier
import pickle

import copy

import time
import os
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def split_and_scale(X: ndarray, y: ndarray, validation_split: float = 0.33) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Split train data into training and validation data, then scale the data.

    Args:
        X (ndarray): dependent variables
        y (ndarray): independent variables
        validation_split (float, optional): % size of validation split. Defaults to 0.33.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: Return xtrain, xvalidation, ytrain, yvalidation
    """
    scaler = StandardScaler()
    xt, xv, yt, yv = train_test_split(X, y, test_size=validation_split)
    return  scaler.fit_transform(xt), scaler.fit_transform(xv), yt, yv 


def runModel(model: ClassifierMixin, x_train: ndarray, x_valid: ndarray, y_train: ndarray, y_valid: ndarray) -> Tuple[float, float, float, float]:

    if model == None:
        raise ValueError("no model supplied")
    if not isinstance(model, ClassifierMixin):
        warn("did you pass a valid sklearn model?")
    
    model.fit(x_train, y_train)
    accuracy_train = model.score(X=x_train, y=y_train)
    accuracy_valid = model.score(X=x_valid, y=y_valid)

    perplex_train = np.inf
    perplex_valid = np.inf

    try:
        probs_train = model.predict_proba(x_train)
        probs_valid = model.predict_proba(x_valid)
        perplex_train = perplexity(y_train, probs_train)
        perplex_valid = perplexity(y_valid, probs_valid)
    except AttributeError:
        warn(f'Model {getNiceModelName(model)} could not predict probabilities')
    
    return getNiceModelName(model), accuracy_train, perplex_train, accuracy_valid, perplex_valid


def runModelCV(model: ClassifierMixin, model_params: dict, x: ndarray, y: ndarray, n_iterations: int=10, k_folds: int=5) -> Tuple[float, float, float, float]:
    per_tr = []
    per_te = []
    acc_tr = []
    acc_te = []
    modelIndices = []
    modelDict = {}
    modelParams = []

    if model == None:
        raise ValueError("no model supplied")
    if not isinstance(model, ClassifierMixin):
        warn("did you pass a valid sklearn model?")
    optimisedParams = optimiseModelParams(model, model_params, x, y )
    if optimisedParams is not None:
        model.set_params(**optimisedParams)
    kfolds = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_iterations)

    j = 0
    for test_idx, train_idx in kfolds.split(x, y):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model.fit(X_train, y_train)
        accuracy_train = model.score(X=X_train, y=y_train)
        accuracy_test = model.score(X=X_test, y=y_test)

        perplex_train = np.inf
        perplex_test = np.inf

        try:
            probs_train = model.predict_proba(X_train)
            probs_test = model.predict_proba(X_test)
            perplex_train = perplexity(y_train, probs_train)
            perplex_test = perplexity(y_test, probs_test)
        except AttributeError:
            warn(f'Model {getNiceModelName(model)} could not predict probabilities')
        
        j += 1
        #save everything
        modelIndex = f'{getNiceModelName(model)}-{j}'
        modelIndices.append(modelIndex)
        per_tr.append(perplex_train)
        per_te.append(perplex_test)
        acc_tr.append(accuracy_train)
        acc_te.append(accuracy_test)

        modelDict[modelIndex] = copy.deepcopy(model)
    print( f'runModelCV call done' )
    model_results = pd.DataFrame({"Train Accuracy": acc_tr, "Train Perplex": per_tr, "Validation Accuracy": acc_te, "Validation Perplex": per_te, "Params": str(optimisedParams)}, index=modelIndices)
    return model_results, modelDict

def optimiseModelParams(model: ClassifierMixin, paramDistributions: dict, x: ndarray, y: ndarray, n_iterations: int=50, k_folds: int=5):
    if isinstance(model, DummyClassifier):
        return None
    kfolds = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_iterations)
    searcher = RandomizedSearchCV(estimator=model, param_distributions=paramDistributions, n_iter = n_iterations, n_jobs=-1, refit=True, cv=kfolds)
    searcher.fit(x,y)
    pickle.dump(pd.DataFrame(searcher.cv_results_), open(f'{getNiceModelName(model)}-{generate_file_id()}.pickle', 'wb'))
    return searcher.best_params_

def perplexity(y, y_prob):
    return np.exp(np.mean(log_loss(y, y_prob)))

def getNiceModelName(m):
    return m.__str__().split('(')[0]


def Read_data_output_class4(filename="npf_train.csv"):
    npf = pd.read_csv(filename)
    npf = npf.set_index("date")
    npf["class4"] = npf["class4"].astype("category")
    npf = npf.drop("id", axis=1)
    npf = npf.drop("partlybad", axis=1)
    y = npf["class4"]
    x = npf.drop("class4", axis=1)

    #encode y
    enc = LabelEncoder()
    y_encoded = enc.fit_transform(y.values)
    return x, y_encoded, enc

def Read_data_output_class2_or_testdata(binary = bool, training_data = bool, filename="npf_train.csv"):
    # Reads data from training or test files, and outputs x and y if the data is for training for class 2
    # y output can be 1 or 0 or strings event/nonevent, if is test data y is 0
    # Example :
    #[x,y] = Read_data_output_class2_or_testdata(binary=True, training_data=True, filename = "npf_train.csv")
    #[x1,y1] = Read_data_output_class2_or_testdata(binary=False, training_data=False, filename = "npf_test_hidden.csv")

    npf = pd.read_csv(filename)
    npf = npf.set_index("date")
    npf["class4"] = npf["class4"].astype("category")
    npf = npf.drop("id",axis=1)
    npf = npf.drop("partlybad",axis=1)
    if training_data:
        class2 = np.array(["event"]*npf.shape[0],dtype="object")
        class2[npf["class4"]=="nonevent"] = "nonevent"
        npf["class2"] = class2
        npf["class2"] = npf["class2"].astype("category")
        y = npf["class2"]
        x = npf.drop("class2", axis=1)
        #this to output 1/0 instead of string event/nonevent
        if binary:
            y_b = pd.get_dummies(y, prefix=['event', "nonevent"])
            y = y_b.iloc[:,0]
            y = y.to_numpy()
    else:
        y = 0
        x = npf
    x = x.drop("class4", axis=1)
    return x,y

def training_with_PCA(x_training, num_PCA=15):
    #outputs PCA
    # example:
    # [x_PCA,pca] = training_with_PCA(x,15)
    # if required all PCA values
    # [x_PCA, pca] = training_with_PCA(x)
    pca = PCA(n_components=num_PCA)
    x_training_PCA = pca.fit_transform(x_training)
    #if len(num_PCA)>0:
    #    num_PCA = int(num_PCA[0])
    #    x_training_PCA = x_training_PCA[:, :num_PCA]
    return x_training_PCA, pca

def split_training_validate(x,y,n):
    index = np.random.choice(len(x), n, replace=False)
    x_training = x[index, :]
    y_training = y[index]
    x_validate = np.delete(x, index, axis=0)
    y_validate = np.delete(y, index, axis=0)
    return x_training, y_training, x_validate, y_validate

def scaling(x, scaler_obj=StandardScaler):
    scaler = scaler_obj()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled

def runModelStepwiseSelection(model: ClassifierMixin, x: ndarray, y: ndarray):
    time_start = time.time()
    
    if model == None:
        raise ValueError("no model supplied")
    if not isinstance(model, ClassifierMixin):
        warn("did you pass a valid sklearn model?")

    classifier_pipeline = make_pipeline(StandardScaler(), model)
    
    cv = KFold(n_splits=10, random_state=0, shuffle=False)

    sfs1 = SFS(classifier_pipeline, 
        k_features=(1, 100), 
        forward=True, 
        scoring='accuracy',
        cv=cv,
        n_jobs=-1
    )

    sfs1.fit(x,y)
    time_end = time.time()
    time_run = time_end - time_start
    print(model, time_run)

    return {"model": model, "features": sfs1.k_feature_idx_, "scores": sfs1.k_score_}


def generate_file_id():
    return "-".join([time.strftime('%d%m%y_%H%M%S'), os.getlogin()])