# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import feature_extraction
import numpy as np
import sklearn.preprocessing as pp

def get_features():
    feature_set1=[]
    feature_set2=[]
    feature_set3=[]
    for grid in range(len(feature_extraction.data)): #get features from feature_extraction python file and zip them into one array X1
        for x in range(1,7):
            for y in range(1,7):
                point = (x,y)
                feature_set1.append(feature_extraction.feature_set_1(point,grid))
                feature_set2.append(feature_extraction.feature_set_2(point,grid))
                feature_set3.append(feature_extraction.feature_set_3(point,grid))
    features = zip(feature_set1 ,feature_set2 , feature_set3)
    X1 =[]
    for j in range(len(features)):
        X1.append(features[j][0] + features[j][1]+ features[j][2])
    return X1
def eveluate_model(grid):
    X1= get_features() #get features
    Y = feature_extraction.getY()
    stdsc = pp.StandardScaler().fit(X1) #standarizing the features
    X_std = stdsc.transform(X1)
    svm = SVC(kernel='rbf', random_state=0, C=1) #SVM modeling
    svm.fit(X_std, Y)
    X_std_grids = np.array([X_std[x:x + 36] for x in xrange(0, len(X_std), 36)]) #getting 5000 grids each has 6*6 grid
    policy_ = svm.predict(X_std_grids[grid]) #predict grid?
    policy={}
    xy =[]
    for x in range(1,7):
        for y in range(1,7):
            xy.append((x,y))
    policy.update(dict(zip(xy, policy_))) #getting policies
    return policy
