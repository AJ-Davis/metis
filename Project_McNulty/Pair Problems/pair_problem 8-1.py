#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 07:38:30 2018

@author: ajdavis
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

np.random.seed(42)

def generate_congress_data(num_congressmen=100):
    votes = [0,1]
    senators = np.random.choice(votes, replace=True, size=(num_congressmen,3))
    df = pd.DataFrame(senators, columns=['vote1','vote2','vote3'])

    def calculate_party(row):
        x = row['vote1']
        y = row['vote2']
        z = row['vote3']

        party = 0.7*x + 0.5*y - z + np.random.normal(0,0.3)
        if party > 0.1:
            return 'Dem'
        elif party > 0.01:
            return 'Ind'
        else:
            return 'Rep'

    df['party'] = df.apply(calculate_party,axis=1)
    print(df.party.value_counts())
    return df.iloc[:,:-1],df.iloc[:,-1]

X, y = generate_congress_data(num_congressmen=400)

X[y=='Dem'].T[0]

for i in X:
   votes = X[y=='Dem'].T[i]


def occurances(outcome):
    no_examples = len(outcome)
    prob = dict(Counter(outcome))
    for key in prob.keys():
        prob[key] = prob[key]/float(no_examples)
    return prob

occurances(y)












def naive_bayes(X, y, new_sample):
    classes = np.unique(y)
    rows, cols = np.shape(X)
    
    likelihoods = {}
    for cls in classes:
        # initialize dict
        likelihoods[cls] = defaultdict(list)
        
    for cls in classes:
        # take samples of one class at a time
        row_indicies = np.where(outcome == cls)[0]
        subset = X[row_indicies, :]
        r, c = np.shape(subset)
        
        for j in range(0, c):
            likelihoods[cls][j] += list(subset[:,j])
            
    for cls in classes:
        for j in range(0, cols):
            likelihood[cls][j] = occurances(likelihoods[cls][j])
            
    results = {}
    
    for cls in classes:
        class_probability = class_probabilities[cls]
        for i in range(0, len(new_sample)):
            relative_values = likelihoods[cls][i]
            if new_sample[i] in relative_values.keys():
                class_probability *= relative_values[new_sample[i]]
            else:
                class_probability *= 0
            results[cls] = class_probability
            
    return results

naive_bayes()
            

