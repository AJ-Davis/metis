#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 07:55:49 2018

@author: ajdavis
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
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
# Import df
df = pd.read_csv('Practice_data.csv')
df.columns

# Transform 
df['ln_Ind_Variable_4'] = np.log(abs(df['Ind_Variable_4']) + 0.00000000001)
df['sqrt_Ind_Variable_4'] = np.sqrt(abs(df['Ind_Variable_4']))
df['cubert_Ind_Variable_4'] = df['Ind_Variable_4']**(1/3)


# Plot dep vs indep vars
pd.scatter_matrix(df)
sns.pairplot(df)

# SciKit
y1 = df[['Dep_Variable']]
X1 = df[['Ind_Variable_1','Ind_Variable_2','Ind_Variable_3',
        'Ind_Variable_4','Ind_Variable_5']]

y2 = df[['Dep_Variable']]
X2= df[['Ind_Variable_1','Ind_Variable_2','Ind_Variable_3',
        'Ind_Variable_4']]

y3 = df[['Dep_Variable']]
X3 = df[['Ind_Variable_1','Ind_Variable_2','Ind_Variable_3',
        'Ind_Variable_4', 'cubert_Ind_Variable_4']]



## Cross-Validation

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3)

# Statsmodels

# Create your model
mod1 = sm.OLS(y_train1,sm.add_constant(X_train1), formula = "Dep_Variable ~ Ind_Variable_1 + Ind_Variable_2 + Ind_Variable_3 + Ind_Variable_4 + Ind_Variable_5")
mod2 = sm.OLS(y_train2,sm.add_constant(X_train2), formula = "Dep_Variable ~ Ind_Variable_1 + Ind_Variable_2 + Ind_Variable_3 + Ind_Variable_4")
mod3 = sm.OLS(y_train3,sm.add_constant(X_train3), formula = "Dep_Variable ~ Ind_Variable_1 + Ind_Variable_2 + Ind_Variable_3 + ln_Ind_Variable_4")



# Fit your model to your training set

mod1.fit().summary()
mod2.fit().summary()
mod3.fit().summary()

lr = LinearRegression()

# Fit the model against the training data
lr.fit(X_train1, y_train1)
# Evaluate the model against the testing data
lr.score(X_test1, y_test1)

lr.fit(X_train2, y_train2)
# Evaluate the model against the testing data
lr.score(X_test2, y_test2)

lr.fit(X_train3, y_train3)
# Evaluate the model against the testing data
lr.score(X_test3, y_test3)









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
