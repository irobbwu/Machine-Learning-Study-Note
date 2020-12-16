import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], 'The size of y_predict must be equal to y_predict!'
    
    return sum(y_true == y_predict) / len(y_predict)