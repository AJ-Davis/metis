#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:20:50 2018

@author: ajdavis
"""

from sklearn.model_selection import cross_val_predict
from sklearn import cross_validation
import matplotlib.pyplot as plt
import pandas as pd
import json
import urllib.request
import numpy as np
import geocoder
import requests 
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import time
import os
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import matplotlib.pylab as plt 
import numpy as np 
from sklearn.grid_search import GridSearchCV

## Look at correlations
weed_df_munge.corr()
sns.heatmap(weed_df_munge.corr(), cmap="seismic");
sns.pairplot(weed_df_munge, size = 1.2, aspect=1.5);


## Predicted vs. Actual
predicted = cross_validation.cross_val_predict(est, X, y, cv=5, n_jobs = 1)


fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])
ax.set_xlabel('Measured $/gram')
ax.set_ylabel('Predicted $/gram')
plt.show()



## Diagnostic Plots

diagnostic_plots(X, y, model_fit=None)
fit.resid.plot(style='o', figsize=(12,8))
diagnostic_plots.diagnostic_plots(cars.drop('price', axis=1), cars['price'], fit1)



## Learning Curves

train_sizes, train_scores, test_scores = learning_curve(lr, X, y, cv= 4)
ave_train_scores = train_scores.mean(axis=1)
ave_test_scores = test_scores.mean(axis=1)

learn_df = pd.DataFrame({
    'train_size': train_sizes,
    'train_score': ave_train_scores,
    'test_score': ave_test_scores
})
learn_df


# plot it!# plot  
plt.plot(learn_df['train_size'], learn_df['train_score'], 'r--o', label='train scores')
plt.plot(learn_df['train_size'], learn_df['test_score'], 'b--x', label='test size')
plt.legend(loc='lower right')
plt.ylim(-1,1)

# Random Forest Diagnostics
plot_overfit(X,y,ensemble.RandomForestRegressor,{'max_features':range(1,15)})
plot_overfit(X,y,ensemble.RandomForestRegressor,{'min_samples_leaf':range(1,15)})
plot_overfit(X,y,ensemble.RandomForestRegressor,{'max_depth':range(1,8)})
plot_overfit(X,y,ensemble.RandomForestRegressor,{'n_estimators':[1,5,10,20,30,50,100,200,300,500,1000]},param_static={'learning_rate':.75})
plot_overfit(X,y,ensemble.RandomForestRegressor,{'n_estimators':[1,5,10,20,30,50,100,200,300,500,1000]})
