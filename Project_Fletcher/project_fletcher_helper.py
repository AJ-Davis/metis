#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 20:18:11 2018

@author: ajdavis
"""

## Load Relevant Libraries

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
import json
import urllib.request
import itertools
from time import sleep
from random import randint
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import fuzzymatcher
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
import os
from selenium import webdriver
import time
from sklearn.cluster import DBSCAN, SpectralClustering, MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import SpectralClustering
from scipy.stats import ttest_ind
from tabulate import tabulate
import time
import warnings
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

## Wikileaf Data

def get_wllinks():
    url = 'https://www.wikileaf.com/strains/'
    response = requests.get(url)
    wl = response.text
    wl_soup = BeautifulSoup(wl,'lxml')
    
    links = []
    
    for div in wl_soup.find_all("div", {"class":"item-title"}):
        for link in div.select("a.ellipsis "):
            links.append(link['href'])
    return links



def get_wldata(links):
    wl_df = pd.DataFrame()
    for l in links:
        texts = []
        effects = []
        values = []
        
        url = l
        response = requests.get(url)
        wl = response.text
        soup = BeautifulSoup(wl,'lxml')
        text = [x.get_text() for x in soup.find_all('p')]
        text = ' '.join(text)
        texts.append(text)
        
        for div in soup.find_all("div", {"class":"strain-bar"}):
            for v in div.select("input"):
                values.append(v['value'])
                
                
        for div in soup.find_all("div", {"class":"strain-bar"}):
            for e in div.select("input"):
                effects.append(e['name'])
        ob = pd.DataFrame(values).T 
        ob.columns = effects
        ob['Text'] = texts
        ob['Strain'] = url.rsplit('/', 2)[-2]
        try:
            wl_df = wl_df.append(ob)
        except Exception:
            print('Shape Error')
    return wl_df


## Cannabis Reports Data

def get_crlinks(search):
    sleep(randint(1,10))
    url = 'https://www.cannabisreports.com/api/v1.0/strains/search/{}'.format(search)
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    return data['data']




def get_creffects(ucpcs):
    sleep(randint(1,10))
    url = 'https://www.cannabisreports.com/api/v1.0/strains/{}/effectsFlavors'.format(ucpcs)
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    return data.values()


## Leafly Data

# Have to click through pop-ups manually initially
def get_leaflinks(): 
    chromedriver = "/Applications/chromedriver" # path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(chromedriver)
    driver.get("https://www.leafly.com/explore/sort-alpha")
    more_button=driver.find_element_by_xpath('.//button[@class="ga_Explore_LoadMore m-button m-button--green m-button--lg"]')
    
    i = 0
    while i < 5:
        try:
            more_button=driver.find_element_by_xpath('.//button[@class="ga_Explore_LoadMore m-button m-button--green m-button--lg"]')
            time.sleep(2)
            more_button.click()
            time.sleep(10)
            i += 1
        except Exception as e:
            print(e)
            break
    print('Complete')
    time.sleep(10)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    links = []
    
    for div in soup.find_all("li", {"class":"grid-1-2 grid-sm-1-4 grid-md-1-6"}):
        for link in div.select("a.ga_Explore_Strain_Tile "):
            links.append(link['href'])

    links_rest = []
    
    for div in soup.find_all("li", {"class":"grid-1-2 grid-sm-1-4 grid-md-1-6 ng-scope"}):
        for link in div.select("a.ng-scope"):
            links_rest.append(link['href'])
            
    all_links = links + links_rest
    driver.close()
    return all_links


def get_leafdata(links):
    texts = []
    for l in links:
        url = 'https://www.leafly.com' + l
        response = requests.get(url)
        wl = response.text
        soup = BeautifulSoup(wl,'lxml')
        text = [x.get_text() for x in soup.find_all('p')]
        text = ' '.join(text[2:])
        texts.append(text)
    return texts


# Get setiment score from reviews and descriptions for recommender feature
def get_sentiment(text_series):
    # Create stop words
    stop = stopwords.words('english')
    # Make everything lowercase
    text = text_series.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Remove punctuation
    ttext = text.str.replace('[^\w\s]','')
    # Remove stop words
    text = ttext.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # Remove common words
    freq = pd.Series(' '.join(text).split()).value_counts()[:10]
    freq = list(freq.index)
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    # Remove rare words
    freq = pd.Series(' '.join(text).split()).value_counts()[-10:]
    freq = list(freq.index)
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    # Spelling Correction
# =============================================================================
#     text_df['Text1'] = text_df['Text1'].apply(lambda x: str(TextBlob(x).correct()))
# =============================================================================
    # Lemmatization
    text = text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    # Get sentiment scores
    return text.apply(lambda x: TextBlob(x).sentiment[0])


# Recommender system function for flask app
def get_similar(strains_dict, n=None):
    """
    calculates which strains are most similar to the input strains. 
    Must not return the strains that were inputted.
    
    Parameters
    ----------
    strains: list
        some strains!
    
    Returns
    -------
    ranked_strains: list
        rank ordered strains
    """
    strains = [strains_dict['Strain1'], strains_dict['Strain2'], strains_dict['Strain3']]
    des_dict = pd.Series(df.Description.values,index=df.TestStrain).to_dict()
    strains = [strain for strain in strains if strain in dists.columns]
    strains_summed = dists[strains].apply(lambda row: np.sum(row), axis=1)
    strains_summed = strains_summed.sort_values(ascending=False)
    ranked_strains = strains_summed.index[strains_summed.index.isin(strains)==False]
    ranked_strains = ranked_strains.tolist()
    recs = ranked_strains[:n]
    descs = [des_dict[k] for k in recs]

        
    return recs + descs




# Loop through multiple clustering algorithms
def do_clustering(df):

    mask = ['CBDmax', 'THCmax']
    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}
    
    datasets = [
        (df[mask], {})]
    
    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
    
        X = dataset
    
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
    
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
    
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
    
        # ============
        # Create cluster objects
        # ============
        
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')
    
        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('Birch', birch),
            ('GaussianMixture', gmm)
        )
    
        for name, algorithm in clustering_algorithms:
            t0 = time.time()
    
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)
    
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                dataset[name] = algorithm.labels_.astype(np.int)
            else:
                dataset[name] = algorithm.predict(X)
        dataset.name = dataset
            
    df_new = datasets[0][0]

    
    df_tab = pd.concat([df, df_new.iloc[:,2:]], axis = 1)
    return df_tab


# Get cluster means
def get_chemovar_means(cdf, algs):
    dfs = []
    for name, algorithm in algs:
        mask = [name, 'CBDmax','THCmax']
        df = cdf[mask]
        data = pd.DataFrame(df.groupby(name).mean())
        dfs.append(data)
    

    
    return dfs


# Look at differences by cluster
def get_pvals(cdf, effects, algs):
    results = []
    for effect in effects:
        for name, algorithm in algs:
            mask = [name, effect]
            df = cdf[mask]
            
            values_per_group = {col_name:col for col_name, col in df.groupby(name)[effect]}
            
            c0_c1 = ttest_ind(values_per_group[0], values_per_group[1])[1]
            c0_c2 = ttest_ind(values_per_group[0], values_per_group[2])[1]
            c1_c2 = ttest_ind(values_per_group[1], values_per_group[2])[1]
            
            result = pd.DataFrame({
                    'Clustering Algorithm': name,
                    'Effect': effect,
                    'C1_C2': [c0_c1],
                    'C1_C3': [c0_c2],
                    'C2_C3': [c1_c2]
                    
                    })
            results.append(result)
    return results




