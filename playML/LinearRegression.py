import numpy as np
from .metrics import r2_score

class LinearRegression:
    
    def __init__(self):
        
        self.coef_ = None
        self.intercept_ = None
        self._theta = None
        
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], 'The size of X_train must be equal to y_train'
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
    def fit_gd(self, X_train, y_train, eta = 0.01, n_iters = 1e4):
        assert X_train.shape[0] == y_train.shape[0], 'The size of X_train must be equal to y_train'
        
        def J(beta, X_b, y):
            try:
                return np.sum((y - X_b.dot(beta)) ** 2) / len(X_b)
            except:
                return float('inf')
        
        def dJ(beta, X_b, y):
            '''
            res = np.empty(len(beta))
            res[0] = np.sum(X_b.dot(beta) - y)
            for i in range(1, len(beta)):
                res[i] = (X_b.dot(beta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)
            '''
            
            return X_b.T.dot(X_b.dot(beta) - y) / len(X_b)
        
        def gradient_descent(X_b, y, initial_beta, eta, n_iters = 1e4, epsilon = 1e-8):

            beta = initial_beta
            i_iter = 0
            
            while i_iter < n_iters:
                gradient = dJ(beta, X_b, y)
                last_beta = beta
                beta = beta - eta * gradient
            
                if (abs(J(beta, X_b, y) - J(last_beta, X_b, y)) < epsilon):
                    break
                
                i_iter += 1
            
            return beta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_beta = np.zeros(X_train.shape[1] + 1)
        self._theta = gradient_descent(X_b, y_train, initial_beta, eta, n_iters)
        
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self
    
    def fit_sgd(self, X_train, y_train, n_iters = 5, t0 = 5, t1 = 50):
        assert X_train.shape[0] == y_train.shape[0], 'The size of X_train must be equal to y_train'
        assert n_iters >=1
        
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2
        
        def sgd(X_b, y, initial_theta, n_iters = 5, t0 = 5, t1 = 50):
            
            def learning_rate(t):
                return t0 / (t + t1)
            
            theta = initial_theta
            m = len(X_b)
            
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient
            
            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_train.shape[1] + 1)
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self
    
    
    def predict(self, X_predict):
        assert self.intercept_ is not None, 'Must fit before predict'
        assert X_predict.shape[1] == len(self.coef_), 'The feature number of X_predict must be equal to X_train'
        
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)
        
        
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        
        return r2_score(y_test, y_predict)
    
    def __repr__(self):
        return 'LinearRegression()'
