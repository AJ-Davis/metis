#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:24:05 2018

@author: ajdavis
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import patsy
from diagnostic_plots import diagnostic_plots
import itertools
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import learning_curve

import warnings
warnings.filterwarnings('ignore')

# Import helper functions
from project_luther_setup import *
# =============================================================================
# TO DO: 
# FIX MISSING THC AND CBD
# ADD A STRAIN FEATURE
# ADD A LOCATION FEATURE
# =============================================================================

## Scrape top strains
top_strains = get_top_strains()

## Scrape strain  THC and CBD info
strain_info = get_strain_info()

    
## Get list of dispensary names
disp_cities = ['San Francisco, CA', 'Seattle, WA', 'Portland, OR', 'Reno, NV',
               'Las Vegas, NV', 'Los Angeles, CA', 'Denver, CO', 'Salem, OR',
               'Eugene, OR', 'Gresham, OR', 'Hillsboro, OR', 'Oakland, CA',
               'Vallejo, CA', 'San Jose, CA', 'Sacramento, CA', 'Boulder, CO',
               'San Diego, CA']
gcs = get_gcs(disp_cities)    
disp_names = list(itertools.chain.from_iterable(list(map(get_disp_names, gcs))))


## Loop through dispensary menus
weed_df = pd.concat(map(menu_to_df, disp_names))

## Pickle
weed_df.to_pickle('weed_df.pkl')

## Munge data
weed_df_munge = munge_df(weed_df)


## Feature Engineering
weed_df_munge['price'].hist()
weed_df_munge['ln_price'].hist()

## Look at correlations
weed_df_munge.corr()
sns.heatmap(weed_df_munge.corr(), cmap="seismic");
sns.pairplot(weed_df_munge, size = 1.2, aspect=1.5);


## Modeling - statsmodels

my_formula = "ln_price ~ THC + CBD + category_name_Sativa + category_name_Indica"
# Create your feature matrix (X) and target vector (y)
y, X = patsy.dmatrices(my_formula, data=weed_df_munge, return_type="dataframe")
# Create your model
mod = sm.OLS(y,X)
# Fit your model to your training set
fit = mod.fit()
fit.summary()

## Diagnostic Plots

diagnostic_plots(X, y, model_fit=None)
fit.resid.plot(style='o', figsize=(12,8))
diagnostic_plots.diagnostic_plots(cars.drop('price', axis=1), cars['price'], fit1)

## Modeling - sklearn
X = weed_df_munge[['THC', 'CBD', 'category_name_Sativa', 'category_name_Indica']]
y = weed_df_munge[['ln_price']]

lr = LinearRegression()
lr.fit(X, y)
lr.score(X, y)


# Let's start with filtering features using p-value:# Let's 
# =============================================================================
# est=LinearRegression()
# from sklearn import feature_selection as f_select
# 
# sig_columns=[]
# pvals=[] # will be the list of all significant columns' p-values
# 
# for feature in X.columns:
#     #get pval on feature by feature basis
#     pval=f_select.f_regression(X[[feature]],y) # gets f-value and p-value
#     print(pval)
#     if pval[1][0]<.02: 
#         sig_columns.append(feature)
#         pvals.append(pval[1][0])
#         
# X_trans=X[sig_columns]
# cross_val_score(est,X_trans,y,cv=5,scoring='r2').mean()
# =============================================================================


## Cross-Validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Fit the model against the training data
lr.fit(X_train, y_train)
# Evaluate the model against the testing data
lr.score(X_test, y_test)

scores = cross_val_score(lr, X, y, cv=10, scoring='mean_squared_error')
print(-scores)

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


    