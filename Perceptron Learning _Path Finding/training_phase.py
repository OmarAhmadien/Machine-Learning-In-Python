# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import feature_extraction
import pylab
import perceptron_classifier

# scatter plot
x = feature_extraction.x

def visualise_features(x):
    f1 = feature_extraction.feature1(x)
    f2 = feature_extraction.feature2(x)
    f11 = f1[:len(f1)/2]
    f12 = f1[len(f1)/2:]
    f21 = f2[:len(f2)/2]
    f22 = f2[len(f2)/2:]
    #plt.plot(f11,f12,"bo",f21,f22,"r+")
    plt.scatter(f11,f21 , c= 'blue',marker='o', s=30, label='Positives')
    plt.scatter(f12,f22 , c= 'red',marker='o', s=30, label='Negatives')
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend(loc='upper left')
    plt.show()
    return
visualise_features(x)

def train_classifier():
    w= perceptron_classifier.Perceptron()
    weights_avg = w.fit(feature_extraction.features, feature_extraction.output)
    print weights_avg
    return weights_avg

def visualize_decision_boundary():

    return
