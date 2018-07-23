#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:19:30 2018

@author: ajdavis
"""
from sklearn import linear_model,ensemble, tree, model_selection, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Need to figure out how to set random seed

weed_df_munge = munge_df(weed_df)

## Feature engineering

# Research for feature engineering
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

# Look at correlations

weed_df_munge.corr()
sns.heatmap(weed_df_munge.corr(), cmap="seismic");
sns.pairplot(weed_df_munge, size = 1.2, aspect=1.5);

## Specifiy Models
# Features
mf1 = ['price', 'ln_price', 'THC2', 'THC3', 'THC4', 'THC:rating']
mf2 = []
mf3 = []

# Targets
mt1 = ['price']
mt2 = []
mt3 = []

def get_features(features_to_exclude):
    return [x for x in weed_df_munge.columns if x not in features_to_exclude]
    
def get_target(target_to_include):
    return [x for x in weed_df_munge.columns if x in target_to_include]
    
features = get_features(mf1)
target = get_target(mt1)

design = target + features
analysis_df = weed_df_munge[design]


## Set X and y

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

## Fit the test data with the best model
est = ensemble.RandomForestRegressor()
est.fit(X_train_scale, y_train)
est.score(X_test_scale,y_test)


## Tune best model

# Random Forest
# Get top features
coefs = est.fit(X_train_scale,y_train).feature_importances_
imp_coefs = sorted(zip(X.columns,coefs), key = lambda x:x[1], reverse=True)
top_features = list(zip(*imp_coefs))[0][:5]

# Subset on top features
X_train_reduced = X_train[list(top_features)]
X_test_reduced = X_test[list(top_features)]

# Re-scale
X_train_reduced_scale = scaler.fit_transform(X_train_reduced)
X_test_reduced_scale = scaler.fit_transform(X_test_reduced)

## Re-score df from training data
scores_reduced = get_scores(X_train_reduced_scale, y_train)
scores_reduced.loc[scores_reduced['Scores'].idxmax()]['Model']

## Re-fit the test data 
est = ensemble.RandomForestRegressor()
est.fit(X_train_reduced_scale, y_train)
est.score(X_test_reduced_scale,y_test)



from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = ensemble.RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_scale, y_train)
rf_random.best_params_

# Evaluate Random Search
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = ensemble.RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train_scale, y_train)
base_accuracy = evaluate(base_model, X_test_scale, y_test)


# Further optimize with grid search

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)







coefs = models['randomForest'].fit(X,y).feature_importances_
sorted(zip(X.columns,coefs), key = lambda x:x[1], reverse=True)

X_reduced = X[['F4','F2','F1','F5','F3',1]]

for name,model in models.items():
    scores = model_selection.cross_val_score(model, X_reduced, y, n_jobs=-1)
    print('Model: '+name)
    print("Score: " + str(np.mean(scores)))
    print()
    
    
# Gradient Boost
param_grid={'n_estimators':[100],#,500,1000], 
            'learning_rate': [0.1,0.05,0.02],# 0.01], 
            'max_depth':[4,6], 'min_samples_leaf':[3,5,9,17], 
            'max_features':[1.0,0.3,0.1] } 


n_jobs=4 

cv,best_est = GradientBooster(param_grid, n_jobs)
    
# check for high-variance/over-fitting
plot_overfit(X,y,ensemble.GradientBoostingRegressor,{'max_features':range(1,15)})
plot_overfit(X,y,ensemble.GradientBoostingRegressor,{'min_samples_leaf':range(1,15)})
plot_overfit(X,y,ensemble.GradientBoostingRegressor,{'max_depth':range(1,8)})
plot_overfit(X,y,ensemble.GradientBoostingRegressor,{'n_estimators':[1,5,10,20,30,50,100,200,300,500,1000]},param_static={'learning_rate':.75})
plot_overfit(X,y,ensemble.GradientBoostingRegressor,{'n_estimators':[1,5,10,20,30,50,100,200,300,500,1000]})

# check learning curver to see if adding for data makes sense
model = ensemble.GradientBoostingRegressor
params = {'learning_rate':[.01,.1,.2,.5,.75,.9,1,1.25,1.5]}
param_static = {'max_depth':4}
plot_overfit(X,y,model,params,param_static=param_static)

model = ensemble.GradientBoostingRegressor
params = {'learning_rate':[.01,.1,.2,.5,.75,.9,1,1.25,1.5]}
param_static = {'max_depth':5}
plot_overfit(X,y,model,params,param_static=param_static)

plot_overfit(X,y,ensemble.GradientBoostingRegressor,{'subsample':np.arange(.1,1,.1)})

model = ensemble.GradientBoostingRegressor
params = {'learning_rate':np.arange(.01,.2,.01)}
param_static = {'max_depth':4,'subsample':.3,'n_estimators':300}
plot_overfit(X,y,model,params,param_static=param_static)


model = ensemble.GradientBoostingRegressor
params = {'subsample':np.arange(.1,1,.1)}
param_static = {'max_depth':4,'learning_rate':.03,'n_estimators':500}
plot_overfit(X,y,model,params,param_static=param_static)


model = ensemble.GradientBoostingRegressor(learning_rate=.03,subsample=.2,n_estimators=500,max_depth=4)

