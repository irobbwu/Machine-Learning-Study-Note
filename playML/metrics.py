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

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))

def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_test, y_log_predict), FP(y_test, y_log_predict)],
        [FN(y_test, y_log_predict), TP(y_test, y_log_predict)]
    ])
    
def precision_score(y_true, y_predict):
    tp = TP(y_test, y_log_predict)
    fp = FP(y_test, y_log_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
    
def recall_score(y_true, y_predict):
    tp = TP(y_test, y_log_predict)
    fn = FN(y_test, y_log_predict)
    # try...except：异常检测；
        # 没有异常，执行 try 后面的语句；
        # 出现异常，执行 except 后面的语句，
    try:
        return tp / (tp + fn)
    except:
        return 0.0
    
def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.
    
def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.