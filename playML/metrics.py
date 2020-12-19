import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], 'The size of y_predict must be equal to y_predict!'
    
    return sum(y_true == y_predict) / len(y_predict)

def mean_squared_error(y_true, y_predict):
    assert len(y_predict) == len(y_true), 'The size of y_true must be equal to that of y_predict'
    
    return np.sum((y_predict - y_true) ** 2) / len(y_predict)

def root_mean_squared_error(y_true, y_predict):
    assert len(y_predict) == len(y_true), 'The size of y_true must be equal to that of y_predict'
    
    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    assert len(y_predict) == len(y_true), 'The size of y_true must be equal to that of y_predict'
    
    return np.sum(np.absolute(y_predict - y_true) / len(y_predict))
    
def r2_score(y_true, y_predict):
    assert len(y_predict) == len(y_true), 'The size of y_true must be equal to that of y_predict'
    
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
