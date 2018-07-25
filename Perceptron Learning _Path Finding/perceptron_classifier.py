# -*- coding: utf-8 -*-

import numpy as np
import random
import feature_extraction
SEED = 0
random.seed(SEED)

class Perceptron(object):
    def __init__(self, eta=0.1, n_iter=1000):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """Fit training data.
            Parameters
            ----------
            X : {array-like}, shape = [n_samples, n_features]
            n_features is the number of features.
            y : array-like, shape = [n_samples]
            Target values.
            Returns
            -------
            self : object
        """
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        data = []
        for _ in range(self.n_iter):
            errors = 0
            count = np.zeros(1+X.shape[1])
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                count += sum(self.w_)
                errors += int(update != 0.0)
            data.append(count)
            count = np.zeros(1+X.shape[1])
            self.errors_.append(errors)
        weights_avg =  [x / 300000 for x in data]
        np.savetxt('weights.txt',weights_avg)
        return weights_avg
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >=0.0, 1, -1)

