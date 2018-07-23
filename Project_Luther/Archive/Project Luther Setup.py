#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:07:50 2018

@author: ajdavis
"""

import json
import urllib.request
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
import numpy as np
import geocoder
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

## Helper functions
def trans_price(row):
    if row['two_grams'] != 0:
        return row['two_grams']/2
    elif row['eighth'] !=0:
        return row['eighth']/3.5 
    elif row['quarter'] !=0:
        return row['quarter']/7
    elif row['half_ounce'] !=0:
        return row['half_ounce']/14
    elif row['ounce'] !=0:
        return row['half_ounce']/28
    else:
        return row['gram']

def get_gcs(list_names):
    gcs = []
    for name in list_names:
        g = geocoder.google(name)
        gc = g.latlng
        gcs.append(gc)
        gcs = list(map(str, gcs))
        gcs = [gc.replace('[', '') for gc in gcs]
        gcs = [gc.replace(']', '') for gc in gcs]
        gcs = [gc.replace(', ', ',') for gc in gcs]
        gcs = [i for i in gcs if i != 'None']
    return gcs

def get_disp_names(gc):
    url = 'https://api-g.weedmaps.com/wm/v2/location?include%5B%5D=regions.listings&latlng={}'.format(gc)
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    lst = data['data']['regions']['dispensary']['listings']
    disp_df = pd.DataFrame()
    for d in lst:
        df = pd.DataFrame.from_dict(d)
        disp_df= disp_df.append(df)
    return list(disp_df['slug'].unique())


def menu_to_df(disp_names):
    url = 'https://api-g.weedmaps.com/wm/web/v1/listings/{}/menu?type=dispensary'.format(disp_names)
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    try:
        m_dict = { k:[d[k] for d in data['categories']] for k in data['categories'][0] }
    except:
        print("Error")
    weed_df = pd.DataFrame()
    try:
        for d in m_dict['items']:
            df = pd.DataFrame.from_dict(d)
            weed_df = weed_df.append(df)
            weed_df = weed_df[['prices', 'body', 'license_type', 'category_name']]
    except:
        print("Error")
    return weed_df

def munge_df(df):
    df[['THC','CBD']] = df['body'].str.split('THC', 1, expand = True)
    df['CBD'] = pd.to_numeric(df['CBD'].str.split('%').str[0].fillna(0), errors = 'coerce')
    df['THC'] = pd.to_numeric(df['THC'].str.split('%').str[0].fillna(0), errors = 'coerce')
    df['prices'] = df['prices'].astype(str)
    df["prices"] = df["prices"].apply(lambda x : dict(eval(x)) )
    tmp = df["prices"].apply(pd.Series )
    df = pd.concat([df, tmp], axis=1)
    df['price'] = df.apply(trans_price, axis=1)
    df = df[['price', 'THC', 'CBD', 'license_type', 'category_name']]
    df['ln_price'] = np.log(df['price'] + .000000001)
    df['sqrt_price'] = np.sqrt(df['price'])
    df = df[(df != 0).all(1)]
    df = df.dropna()
    df = pd.get_dummies(df)
    return df
