#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:24:05 2018

@author: ajdavis
"""
# =============================================================================
# Set-up
# =============================================================================

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import patsy
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
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import random
from fuzzywuzzy import process, fuzz
from sklearn import linear_model,ensemble, tree, model_selection, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Import helper functions
from project_luther_setup import *
from diagnostic_plots import diagnostic_plots # might not work - may need to run file

# =============================================================================
# Data Processing
# =============================================================================
    
## Get list of dispensary names
disp_cities = ['San Francisco, CA', 'Seattle, WA', 'Portland, OR', 'Reno, NV',
               'Las Vegas, NV', 'Los Angeles, CA', 'Denver, CO', 'Salem, OR',
               'Eugene, OR', 'Gresham, OR', 'Hillsboro, OR', 'Oakland, CA',
               'Vallejo, CA', 'San Jose, CA', 'Sacramento, CA', 'Boulder, CO',
               'San Diego, CA', 'Spokane, WA', 'Bend, OR', 
               'British Columbia, Canada', 'Toronto, Canada', 'Washington D.C.',
               'Winslow, AZ', 'Tacoma, WA', 'Vancouver, WA', 'Bellevue, WA',
               'Kent, WA', 'Everett, WA', 'Renton, WA', 'Federal Way, WA',
               'Yakima, WA', 'Fresno, CA', 'Long Beach, CA', 'Santa Ana, CA',
               'Anaheim, CA', 'Beaverton OR', 'Medford, OR', 'Springfield, OR',
               'Corvallis, OR', 'Carson City, NV', 'Sparks, NV']

# Get city geocodes
gcs = get_gcs(disp_cities) 

# Create a list of dispensary names to pull menu data from   
disp_names = list(itertools.chain.from_iterable(list(map(get_disp_names, gcs))))

## Loop through dispensary menus and create aggregated data set
weed_df = pd.concat(map(menu_to_df, disp_names))

## Pickle
weed_df.to_pickle('weed_df.pkl')

## Munge data
weed_df_munge = munge_df(weed_df)




# =============================================================================
# Feature Engineering
# =============================================================================
# logs, polynomials, interaction terms

# Check to see if target is normally distributed
weed_df_munge['price'].hist()

# Log Transformations
weed_df_munge['ln_price'] = np.log(weed_df_munge['price'] + .000000001) # Add small constant in case of 0s
weed_df_munge['ln_price'].hist()


# Polnomial Transformations
weed_df_munge['THC2'] = weed_df_munge['THC']**2
weed_df_munge['THC3'] = weed_df_munge['THC']**3
weed_df_munge['THC4'] = weed_df_munge['THC']**4


# Interactions
weed_df_munge['THC:rating'] = weed_df_munge['THC']*weed_df_munge['rating']

# Create categorical dummies
weed_df_munge = weed_df_munge.dropna()
weed_df_munge = pd.get_dummies(weed_df_munge)


# =============================================================================
# Modeling 
# =============================================================================


# =============================================================================
# Smaller set of features
# Add polynomials
# Check functional forms
# More features
# More data
# Regularization
# Hyperparameter tuning
# =============================================================================

## Specifiy Models
# Features (add those that should be excluded)
mf1 = ['price', 'ln_price', 'THC2', 'THC3', 'THC4', 'THC:rating']
mf2 = ['price', 'ln_price', 'THC2', 'THC3', 'THC4', 'THC:rating']
mf3 = ['price', 'ln_price', 'THC3', 'THC4', 'THC:rating']

# Targets
mt1 = ['price']
mt2 = ['ln_price']
mt3 = ['price']

# Set X and y 
features = get_features(mf3)
target = get_target(mt3)

design = target + features
analysis_df = weed_df_munge[design]

# Clean categorical dummy column names for patsy (need to write helper function)
analysis_df.columns = analysis_df.columns.str.replace(" - ", "_")
analysis_df.columns = analysis_df.columns.str.replace("-", "_")
analysis_df.columns = analysis_df.columns.str.replace(" ", "_")
analysis_df.columns = analysis_df.columns.str.replace("'", "")
analysis_df.columns = analysis_df.columns.str.replace(".", "")
analysis_df.columns = analysis_df.columns.str.replace("|", "_")
analysis_df.columns = analysis_df.columns.str.replace("&", "_")
analysis_df.columns = analysis_df.columns.str.replace(")", "")
analysis_df.columns = analysis_df.columns.str.replace("(", "")
analysis_df.columns = analysis_df.columns.str.replace("/", "")
analysis_df.columns = analysis_df.columns.str.strip()

# Create formula for patsy
my_formula = analysis_df.columns[0] + " ~ " + \
" + ".join(list(analysis_df.columns[1:len(analysis_df.columns)])) + ' - 1'

# Create your feature matrix (X) and target vector (y)
y, X = patsy.dmatrices(my_formula, data=analysis_df, return_type="dataframe")

# Look at OLS before testing all models
sm.OLS(y, sm.add_constant(X)).fit().summary()

## Test/Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## Normalization
scaler = preprocessing.StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

## Create score df from training data
scores = get_scores(X_train_scale, y_train)
scores.loc[scores['Scores'].idxmax()]['Model']

## Fit the test data with the best model - need to streamline from above
est = ensemble.RandomForestRegressor()
est.fit(X_train_scale, y_train)
est.score(X_test_scale,y_test)





    


# =============================================================================
# TO DO: 
# FIX MISSING THC AND CBD
# ADD A POPULAR STRAIN DUMMY
# ADD RARE STRAIN DUMMY
# BRANDING
# =============================================================================



## Get ancillary Strain Data (this wasn't used in the final )

top_strains = get_top_strains() # Scrape top strains

# Get additional strain  THC and CBD info to supplement missing
strain_info = get_strain_info() # Scrape additional strain info from weedmaps

# Load ancillary data from 
cc_df = pd.read_csv('cannabinoid_content.csv', encoding='iso-8859-1')
cc_df = cc_df.groupby('MatchedName', as_index = False)['THCmax'].agg(np.mean)

# format strain names for matching
missing = weed_df[weed_df['THC'].isnull()]
missing['MatchedName'] = missing['name'].str.upper()
missing['MatchedName'] = missing['MatchedName'].str.replace('[^A-Za-z0-9]+', '')

results = missing['MatchedName'].apply(
        fuzzy_match,
        args = (cc_df['MatchedName'],
                fuzz.token_set_ratio, 70))