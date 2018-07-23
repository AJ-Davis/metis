#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 07:45:06 2018

@author: ajdavis
"""
import numpy as np
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

df = pd.DataFrame(np.random.randint(0, 1000, size = (200, 20)))
Y = df.iloc[:, 0]
X = df.iloc[:, 1:]
mod1 = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))

def get_Rsquared(features):
    df = pd.DataFrame(np.random.randint(0, 1000, size = (200, features)))
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    df.columns = ['X' + str(col) for col in df.columns]
    df = df.rename(columns={'X0':'Y'})
    all_columns = "+".join(df.columns.difference(["Y"]))
    my_formula = "Y~" + all_columns
    # Create your feature matrix (X) and target vector (y)
    y, X = patsy.dmatrices(my_formula, data=df, return_type="dataframe")
    # Create your model
    mod2 = sm.OLS(y,X)
    # Fit your model to your training set
    fit = mod2.fit()
    # Print summary statistics of the model's performance
    return fit.rsquared

Rsq = [get_Rsquared(21), get_Rsquared(41), get_Rsquared(61), get_Rsquared(81), get_Rsquared(101)]
features = [20, 40, 60, 80, 100]
plot_df = pd.DataFrame({'FEATURES': features,
                        'R-Squared': Rsq})
    plot_df.plot(x = 'FEATURES', y = 'R-Squared')

def get_adjRsquared(features):
    df = pd.DataFrame(np.random.randint(0, 1000, size = (200, features)))
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    df.columns = ['X' + str(col) for col in df.columns]
    df = df.rename(columns={'X0':'Y'})
    all_columns = "+".join(df.columns.difference(["Y"]))
    my_formula = "Y~" + all_columns
    # Create your feature matrix (X) and target vector (y)
    y, X = patsy.dmatrices(my_formula, data=df, return_type="dataframe")
    # Create your model
    mod2 = sm.OLS(y,X)
    # Fit your model to your training set
    fit = mod2.fit()
    # Print summary statistics of the model's performance
    return fit.rsquared_adj

adjRsq = [get_adjRsquared(21), get_adjRsquared(41), get_adjRsquared(61), get_adjRsquared(81), get_adjRsquared(101)]
adj_plot_df = pd.DataFrame({'FEATURES': features,
                        'Adjusted R-Squared': adjRsq})
adj_plot_df.plot(x = 'FEATURES', y = 'Adjusted R-Squared')




