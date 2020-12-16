from math import sqrt
from collections import Counter
import numpy as np
from .metrics import accuracy_score
'''from .metrics 寻找当前程序包里的文件'''

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
        y_predict = [self._predict(i) for i in x]
        return np.array(y_predict)

    def _predict(self, x):
        distance = [sqrt(np.sum((i - x) ** 2)) for i in self.x_train]
        nearest = np.argsort(distance)
        top_y = [self.y_train[i] for i in nearest[:self.k]]
        y_predict = Counter(top_y).most_common()[0][0]
        return y_predict
    
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    
    def __repr__(self):
        return 'kNN(k = %d)'%self.k



