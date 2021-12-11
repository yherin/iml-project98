
from sklearn.base import ClassifierMixin
import numpy as np

class DummyClassifier(ClassifierMixin):
    
    

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        return self
    
#0: 0.24, 1: 0.06, 2: 0.1, 3: 0.6

    def predict_proba(self, X):
        return np.array([0.24, 0.06, 0.1, 0.6])

    def score(self, X, y):
        return (y == 3).mean() #dummy classifier always predicts nonevent

    def set_params(self, args):
        pass