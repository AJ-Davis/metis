#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:07:50 2018

@author: ajdavis
"""
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

## Helper functions

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
    disp_df = pd.DataFrame(
            {'dname':[data['listing']['name']],
             'rating':[data['listing']['rating']],
             'region':[data['listing']['region']],
             'zip':[data['listing']['zip_code']],
             
            })
    try:
        m_dict = { k:[d[k] for d in data['categories']] for k in data['categories'][0] }
    except:
        print("Error")
    flowers = ['Indica', 'Sativa', 'Hybrid']
    weed_df = pd.DataFrame()
    try:
        for d in m_dict['items']:
            df = pd.DataFrame.from_dict(d)
            weed_df = weed_df.append(df)
            weed_df = weed_df[['prices', 'body', 'license_type', 
                               'category_name', 'name', 'listing_name']]
            weed_df = weed_df[weed_df['category_name'].isin(flowers)]
            weed_df['THC'] = weed_df['body'].astype(str).str.extract("(\d+.\d+)%").astype(float) 
    except:
        print("Error")
        
    try:
        disp_df = pd.concat([disp_df]*len(weed_df), ignore_index=True)
        disp_df.reset_index(inplace=True, drop=True)
        weed_df.reset_index(inplace=True, drop=True)
        weed_df = pd.concat([weed_df, disp_df], axis=1)
    except:
        print("Error")
    return weed_df


def munge_df(df):

    df['prices'] = df['prices'].astype(str)
    df["prices"] = df["prices"].apply(lambda x : dict(eval(x)) )
    tmp = df["prices"].apply(pd.Series )
    df = pd.concat([df, tmp], axis=1)
    df['two_grams'] = (df['two_grams']/2).replace(0, np.nan)
    df['eighth'] = (df['eighth']/3.5).replace(0, np.nan)
    df['quarter'] = (df['quarter']/7).replace(0, np.nan)
    df['half_ounce'] = (df['half_ounce']/14).replace(0, np.nan)
    df['ounce'] = (df['ounce']/28).replace(0, np.nan)
    df['price'] = df[['gram', 'two_grams', 'eighth', 'quarter', 'half_ounce', 'ounce']].mean(axis=1)
    
# =============================================================================
#     df['name'] = df['name'].str.upper()
#     df['name'] = df['name'].str.replace('[^A-Za-z0-9]+', '')
# =============================================================================
    
    
    df = df[['price', 'THC', 'license_type', 'category_name', 'region', 'rating']]
    df = df[(df != 0).all(1)]

    return df


def get_top_strains():
    url = 'http://cannabis.net/blog/opinion/15-most-popular-cannabis-strains-of-all-time'
    response = requests.get(url)
    print(response.status_code)
    page = response.text
    soup = BeautifulSoup(page,"html5lib")   
    text = [x.get_text() for x in soup.find_all('a', {'href': re.compile(r'https://cannabis.net/strains/')})] 
    return list(filter(None, text))

def get_strain_info():
    chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(chromedriver)
    driver.get("https://weedmaps.com/strains")
    more_button=driver.find_element_by_xpath('.//a[@class="btn btn-more-strains"]')

    while True:
        try:
            more_button=driver.find_element_by_xpath('.//a[@class="btn btn-more-strains"]')
            time.sleep(2)
            more_button.click()
            time.sleep(10)
        except Exception as e:
            print(e)
            break
    print('Complete')
    time.sleep(10)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    text = [x.get_text().strip() for x in soup.find_all('div', class_='strain-cell Hybrid')] 
    text = [i.replace('\n', '') for i in text]
    text = [re.sub('  +', ',', i) for i in text]
    d = {}
    for b in text:
        i = b.split(',')
        x = i[2].split('THC')
        try:
            x[1] = x[1].replace('CBD', '')
        except:
            print('Error')
        try:
            d[i[0]] = float(x[0].strip().replace('%', ''))
# =============================================================================
#             d[i[0]] = [i[1], x[0].strip(), x[1].strip()]
# =============================================================================
        except:
            print('Error')
    driver.quit()
    return d


def plot_overfit(X,y,model_obj,param_ranges,param_static=None): 
    for parameter,parameter_range in param_ranges.items():
        avg_train_score, avg_test_score = [],[]
        std_train_score, std_test_score = [],[]
        
        for param_val in parameter_range:
            param = {parameter:param_val}
            if param_static:
                param.update(param_static)
            
                
            model = model_obj(**param)
            
            train_scores,test_scores = [],[]
            for i in range(5):
                X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = .3)
                model.fit(X_train,y_train)
                
                train_scores.append(model.score(X_train,y_train))
                test_scores.append(model.score(X_test,y_test))
            
            avg_train_score.append(np.mean(train_scores))
            avg_test_score.append(np.mean(test_scores))
            
            std_train_score.append(np.std(train_scores))
            std_test_score.append(np.std(test_scores))
            
        fig,ax = plt.subplots()
        ax.errorbar(parameter_range,avg_train_score,yerr=std_train_score,label='training score')
        ax.errorbar(parameter_range,avg_test_score,yerr=std_test_score,label='testing score')
        
        ax.set_xlabel(parameter)
        ax.set_ylabel('score')
        ax.legend(loc=0)  
        
def fuzzy_match(x, choices, scorer, cutoff):
    results = process.extractOne(x, choices = choices, scorer = scorer, score_cutoff = cutoff)
    results = results.values.tolist()
    results = [x for x in results if x is not None]
    return pd.DataFrame(results1, columns=['strain', 'score', 'index'])


def get_scores(X, y, folds = 5, alpha = 0.5, scoring = 'r2'): 
    
    models = {}
    parameters = {}
    
    models['linear_model'] = linear_model.LinearRegression()
    models['ridge_model'] = linear_model.Ridge()
    models['lasso_model'] = linear_model.Lasso(alpha=alpha)
    models['robust_regression'] = linear_model.SGDRegressor(loss='huber',max_iter=2000)
    models['eps_insensitive'] = linear_model.SGDRegressor(loss='epsilon_insensitive',max_iter=2000)
    models['cart'] = tree.DecisionTreeRegressor(max_depth=7)
    models['extratrees'] = tree.ExtraTreeRegressor(max_depth=7)
    models['randomForest'] = ensemble.RandomForestRegressor()
    models['adaboostedTrees'] = ensemble.AdaBoostRegressor()
    models['gradboostedTrees'] = ensemble.GradientBoostingRegressor()
    
    score_list = []
    for name,model in models.items():
        scores = model_selection.cross_val_score(model, X, y, 
                                                 cv = folds, n_jobs=1,
                                                 scoring = scoring)  
        score_list.append(np.mean(scores))
    scores_df = pd.DataFrame({'Model': list(models.keys()),
                             'Scores': score_list})
    
    return scores_df
    


def GradientBooster(param_grid, n_jobs): 
    estimator = ensemble.GradientBoostingRegressor()
    cv = ShuffleSplit(X_train_scale.shape[0], n_iter=10, test_size=0.2) 
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs) 
    classifier.fit(X_train_scale, y_train_scale)
    print("Best Estimator learned through GridSearch") 
    print(classifier.best_estimator_) 
    return cv, classifier.best_estimator_ 

def get_features(features_to_exclude):
    return [x for x in weed_df_munge.columns if x not in features_to_exclude]
    
def get_target(target_to_include):
    return [x for x in weed_df_munge.columns if x in target_to_include]