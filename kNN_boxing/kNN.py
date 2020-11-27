from math import sqrt
from collections import Counter
import numpy as np

def kNN_classify(k, X_train, y_train, x):
    assert X_train.shape[0] == y_train.shape[0], \
    '自变量的值要和，因变量的维度相同'
    assert 1 <= k <= X_train.shape[0], \
    'k的取值会大于一小于变量格式'
    assert X_train.shape[1] == x.shape[0],\
    '保证X_train中的单个数据，和x是同纬度的'
    distances = [sqrt(np.sum((i - x) ** 2)) for i in X_train]
    nearest = np.argsort(distances)
    top_K = [y_train[i] for i in nearest[:k]]
    return Counter(top_K).most_common()[0][0]
    


