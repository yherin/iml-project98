from __future__ import annotations
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import ClassifierMixin
from warnings import warn


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
        perplex_train = perplexity(y_train, probs_train[:,0])
        perplex_valid = perplexity(y_valid, probs_valid[:,0])
    except AttributeError:
        warn(f'Model {getNiceModelName(model)} could not predict probabilities')

    return getNiceModelName(model), accuracy_train, perplex_train, accuracy_valid, perplex_valid

def perplexity(y, y_prob):
    return np.exp(-1 * (np.mean(np.log(np.where(y == 1, y_prob, 1-y_prob)))))


def getNiceModelName(m):
    return m.__str__().split('(')[0]