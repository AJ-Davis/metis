#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 07:38:43 2018

@author: ajdavis
"""
import numpy as np
import pandas as pd
data = np.arange(0.0, 10.0, 0.1)
from scipy.optimize import fmin
def least_squared(lst):
    result = []
    for n in lst:
        sum_sq_diff = ((2-n)**2)+((7-n)**2)+((1-n)**2)+((5-n)**2)+((10-n)**2)
        result.append(sum_sq_diff)
    df = pd.DataFrame({'Candidate': lst,
                       'Result': result})
    df = df.loc[df['Result'].min()]
    return df

least_squared(data)


def plot_df(lst):
    result = []
    for n in lst:
        sum_sq_diff = ((2-n)**2)+((7-n)**2)+((1-n)**2)+((5-n)**2)+((10-n)**2)
        result.append(sum_sq_diff)
    df = pd.DataFrame({'Candidate': lst,
                       'Result': result})
    return df

least_squared(data)
df = plot_df(data)
df.plot(x = 'Candidate', y = 'Result')


f = lambda n: ((2-n)**2)+((7-n)**2)+((1-n)**2)+((5-n)**2)+((10-n)**2)
fmin(f,data([0,0]))
