# -*- coding: utf-8 -*-
import learning_phase
import feature_extraction
import numpy as np
import sklearn.preprocessing as pp
from sklearn.ensemble import RandomForestClassifier

#RandomForesets Model

def eveluate_model(grid):
    """For a given grid, return the policy for that grid
       Parameters
       ----------
       grid: 2 dimensional array representing a grid
       Return:
       policy: type-dictionary, for states (1,1) up to (6,6)
       """
    Y = feature_extraction.getY()
    X2 = learning_phase.get_features()  # get features
    stdsc = pp.StandardScaler().fit(X2)  # standarizing the features
    X_std = stdsc.transform(X2)
    forest = RandomForestClassifier(n_estimators=10000, #using random forest
                                    random_state=0,
                                    n_jobs=-1)
    forest.fit(X_std, Y)

    X_std_grids = np.array([X_std[x:x + 36] for x in xrange(0, len(X_std), 36)])  # getting 5000 grids each has 6*6 grid
    policy_ = forest.predict(X_std_grids[grid])  # predict grid?
    policy = {}
    xy = []
    for x in range(1, 7):
        for y in range(1, 7):
            xy.append((x, y))
    policy.update(dict(zip(xy, policy_)))  # getting policies
    return policy
