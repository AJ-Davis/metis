#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 21:38:10 2018

@author: ajdavis
"""

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import GridSearchCV

l = Lasso()

df = pd.read_csv('/Users/ajdavis/github/sf18_ds11/class_lectures/week03-luther2/03-assumptions_bayes/Lasso_practice_data.csv')

y = df.iloc[:, 20]
X = df.iloc[:, 0:20]
kf = KFold(2000, 5, Shuffle = True)

alphas_to_test = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]

all_scores=[]

# Instead of using cross_val_score, let's use sklearn.cross_validation.KFold; this will allow
# us to 'manipulate' our training set, 

# get indices of corresponding train & test
for train,test in kf:
    x_train=X.iloc[train]
    y_train=y.iloc[train]
    x_test=X.iloc[test]
    y_test=y.iloc[test]
    pvals=[]
    sig_cols=[]
    
    for feature in x_train.columns:
        pval=f_select.f_regression(x_train[[feature]],y_train)
        if pval[1][0]<.02: 
            sig_cols.append(feature)
            pvals.append(pval[1][0])
            
    l.fit(x_train[sig_cols],y_train)
    r_2=est.score(x_test[sig_cols],y_test)
    all_scores.append(r_2)
        
np.mean(all_scores)
    
alphas_to_test[1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]


for l in all_scores:
    for e in l:
        




# cheating way
grid = GridSearchCV(estimator = l, param_grid=dict(alpha = alphas_to_test), cv = 5)
grid.fit(X, y)
print(grid)
print(grid.best_estimator_.alpha)
print(grid.best_score_)
