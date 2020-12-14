from math import sqrt
from collections import Counter
import numpy as np

class KNNClassifier:
    
    def __init__(self, k):
        self.k = k
        self.x = None
        self.y = None
        
    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0],'The size of x_train must be equal to the size of y_train'
        assert self.k <= x_train.shape[0], 'The size of x_train should be larger than k'
        self.x_train = x_train
        self.y_train = y_train
        return self
    
    
    def predict(self, x):
        assert self.x_train is not None and self.y_train is not None, "must fit before predict !"
        assert x.shape[1] == self.x_train.shape[1], "the feature number of X_predict must be equal to х_ train"
        y_predict = [self._predict(x) for i in x]
        return np.array(y_predict)

    def _predict(self, x):
        self.x = x
        self.distance = [sqrt(np.sum((i - x) ** 2)) for i in self.x_train]
        self.nearest = np.argsort(self.distance)
        self.top_y = [self.y_train[i] for i in self.nearest[:self.k]]
        self.y_predict = Counter(self.top_y).most_common()[0][0]
        return self.y_predict
    
    def __repr__(self):
        return 'kNN(k = %d)'%self.k

