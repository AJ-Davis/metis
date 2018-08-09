#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:28:58 2018

@author: ajdavis
"""
import math
import numpy as np
from scipy import stats

X_train = [[1, 1, 1],
           [0, 0, 0],
           [-1, -1, -1],
           [10, 10, 10]]

y_train = ['red',
           'white',
           'blue',
           'chartreuse']

X_test = [[-1.1, -1.1, -1.1], [-1.1, -1.1, -1.1]]

def calcDist(l1, l2):
    #dist = math.sqrt((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2 + (l1[2]-l2[2])**2)
    
    dist = np.sqrt(np.sum([(l1[i] - l2[i]) ** 2 for i in range(len(l1))]))
    return dist

def wookie_knn(X_train, y_train, X_test, k):
    output = []
    for l1 in X_test:
        dists = []
        for l2 in X_train:
            d = calcDist(l1, l2)
            dists.append(d)
            
        ind = sorted([(i, dists.index(i)) for i in dists], key = lambda x: x[0])
        ind = ind[0:k]
        ind = [y_train[i[1]] for i in ind]
        output.append(ind)
        output = [stats.mode(i) for i in output]
        
    return output

result = wookie_knn(X_train, y_train, X_test, 5)


