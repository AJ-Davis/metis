#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:24:05 2018

@author: ajdavis
"""

# =============================================================================
# Data Processing
# =============================================================================
    
## Get list of dispensary names
url = 'https://raw.githubusercontent.com/grammakov/USA-cities-and-states/master/us_cities_states_counties.csv'
cities = pd.read_csv(url, sep = '|', error_bad_lines=False)
legal_states = ['California', 'Colorado', 'Nevada', 'Oregon', 'Washington']

legal_cities = cities[cities['State full'].isin(legal_states)]
legal_cities = list(legal_cities['City alias'] + ',' + ' ' + legal_cities['State short'])

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
               'Corvallis, OR', 'Carson City, NV', 'Sparks, NV', 'Phoenix, AZ',
               'Chicago, IL', 'Detroit, MI', 'Lansing, MI', 'Durango, CO', 
               'Pueblo, NM', 'Pueblo, CO']

all_cities = legal_cities + disp_cities

# Get city geocodes
gcs = get_gcs(all_cities) 
with open(r'C:\Users\family3340\metis\Project_Luther\gcs.pkl', 'wb') as f:pickle.dump(gcs, f)

with open(r'C:\Users\family3340\metis\Project_Luther\gcs.pkl', 'rb') as f:gcs = pickle.load(f)

# Create a list of dispensary names to pull menu data from   
disp_names = list(itertools.chain.from_iterable(list(map(get_disp_names, gcs))))
with open(r'C:\Users\family3340\metis\Project_Luther\disp_names.pkl', 'wb') as f:pickle.dump(disp_names, f)

with open(r'C:\Users\family3340\metis\Project_Luther\disp_names.pkl', 'rb') as f:disp_names = pickle.load(f)
    



## Loop through dispensary menus and create aggregated data set
weed_df1 = pd.concat(map(menu_to_df, disp_names[0:5000]))
weed_df2 = pd.concat(map(menu_to_df, disp_names[5001:8000]))
weed_df3 = pd.concat(map(menu_to_df, disp_names[8001:10000]))
weed_df3.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df3.pkl')
weed_df4 = pd.concat(map(menu_to_df, disp_names[10001:13000]))
weed_df4.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df4.pkl')
weed_df5 = pd.concat(map(menu_to_df, disp_names[13001:15000]))
weed_df5.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df5.pkl')
weed_df6 = pd.concat(map(menu_to_df, disp_names[15001:18000]))
weed_df6.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df6.pkl')
weed_df7 = pd.concat(map(menu_to_df, disp_names[18001:21176]))
weed_df7.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df7.pkl')

## Pickle
weed_df1.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df1.pkl')
weed_df2.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df2.pkl')
weed_df3.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df3.pkl')

weed_df1 = pd.read_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df1.pkl')
weed_df2 = pd.read_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df2.pkl')
weed_df3 = pd.read_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df3.pkl')

weed_df = pd.concat([weed_df1, weed_df2, weed_df3, weed_df4, weed_df5, weed_df6,
                     weed_df7])

weed_df.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df.pkl')
weed_df = pd.read_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df.pkl')
## Munge data
weed_df_munge = munge_df(weed_df)
weed_df_munge.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df_munge.pkl')
weed_df_munge = pd.read_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df_munge.pkl')

## Average prices by location

# prices by zip
price_zip = (weed_df_munge
.groupby(['zip'])
.agg({
     'price': ['mean', 'count']
 }))
price_zip.columns = price_zip.columns.get_level_values(0)
price_zip.columns = ['price', 'count']
price_zip = (price_zip.loc[price_zip['count'] > 1000])
price_zip.to_csv(r'C:\Users\family3340\metis\Project_Luther\price_zip.csv')

# prices by region
price_region = (weed_df_munge
.groupby(['region'])
.agg({
     'price': ['mean', 'count']
 }))

price_region.columns = price_region.columns.get_level_values(0)
price_region.columns = ['price', 'count']
price_region = (price_region.loc[price_region['count'] > 1000])
price_region.to_csv(r'C:\Users\family3340\metis\Project_Luther\price_region.csv')

price_region.index.name = ''

ax = price_region['price'].sort_values().tail(15).plot.barh(title='$/Gram by Region')
  # s is an instance of Series
fig = ax.get_figure()
fig.savefig(r'C:\Users\family3340\metis\Project_Luther\high_price.jpg', bbox_inches = 'tight')

ax = price_region['price'].sort_values(ascending=False).tail(15).plot.barh()
fig = ax.get_figure()
fig.savefig(r'C:\Users\family3340\metis\Project_Luther\low_price.jpg', bbox_inches = 'tight')

ax = weed_df_munge['category_name'].value_counts().sort_values().plot.pie(title='Frequency of Sub Types', autopct='%1.0f%%')
ax.set_ylabel('')
fig = ax.get_figure()
fig.savefig(r'C:\Users\family3340\metis\Project_Luther\subtype.jpg', bbox_inches = 'tight')

ax = weed_df_munge['dname'].value_counts().sort_values().tail(15).plot.barh(title='Dispensaries with the Most Data')
fig = ax.get_figure()
fig.savefig(r'C:\Users\family3340\metis\Project_Luther\dispensary.jpg', bbox_inches = 'tight')
# =============================================================================
# Feature Engineering
# =============================================================================


weed_df_munge_feat = ts_feature(weed_df_munge)
weed_df_munge_feat.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df_munge_feat.pkl')
weed_df_munge_feat = pd.read_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df_munge_feat.pkl')

# average price per top strain
price_ts = (weed_df_munge_feat.loc[weed_df_munge_feat['Match Score'] > 90])
price_ts = (price_ts
.groupby(['Matched Strain'])
.agg({
     'price': ['mean', 'count']
 }))

## Get ancillary Strain Data
con1 = content_feature(weed_df_munge_feat.iloc[0:100000])
con1.to_pickle(r'C:\Users\family3340\metis\Project_Luther\con1.pkl')
con2 = content_feature(weed_df_munge_feat.iloc[100001:200000])
con2.to_pickle(r'C:\Users\family3340\metis\Project_Luther\con2.pkl')
con3 = content_feature(weed_df_munge_feat.iloc[200001:500000])
con3.to_pickle(r'C:\Users\family3340\metis\Project_Luther\con3.pkl')
con4 = content_feature(weed_df_munge_feat.iloc[500001:701530])
con4.to_pickle(r'C:\Users\family3340\metis\Project_Luther\con4.pkl')

weed_df_munge_feat_con = pd.concat([con1, con2, con3, con4])
weed_df_munge_feat_con.to_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df_munge_feat_con.pkl')
weed_df_munge_feat_con = pd.read_pickle(r'C:\Users\family3340\metis\Project_Luther\weed_df_munge_feat_con.pkl')

weed_df_munge_feat_con = weed_df_munge_feat_con.loc[weed_df_munge_feat_con['Match Score'] >= 100]

# logs, polynomials, interaction terms

# Check to see if target is normally distributed
weed_df_munge_feat_con['price'].hist()

# Log Transformations
weed_df_munge_feat_con['ln_price'] = np.log(weed_df_munge_feat_con['price']) 
weed_df_munge_feat_con['ln_price'].hist()


# Polnomial Transformations
weed_df_munge_feat_con['THC2'] = weed_df_munge_feat_con['THC']**2
weed_df_munge_feat_con['THC3'] = weed_df_munge_feat_con['THC']**3
weed_df_munge_feat_con['THC4'] = weed_df_munge_feat_con['THC']**4


# Interactions
weed_df_munge_feat_con['THC:rating'] = weed_df_munge_feat_con['THC']*weed_df_munge_feat_con['rating']

# Create categorical dummies

dum1 = pd.get_dummies(weed_df_munge_feat_con['category_name'], drop_first=True)
dum2 = pd.get_dummies(weed_df_munge_feat_con['license_type'], drop_first=True)
dum3 = pd.get_dummies(weed_df_munge_feat_con['region'], drop_first=True)

a_df = pd.concat([weed_df_munge_feat_con, dum1], axis=1)
a_df = pd.concat([a_df, dum2], axis=1)
a_df = pd.concat([a_df, dum3], axis=1)


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
mf1 = [
       'price', 'ln_price', 'THC', 'THC2', 'THC3', 'THC4', 'THC:rating', 
       'body', 'category_name', 'dname', 'license_type', 'listing_name',
       'name', 'prices', 'region', 'zip', 'gram', 'Strain Name', 'MatchedName',
       'Match', 'Matched Strain', '_Index', 'Match Score', 'MatchNew'
       ]
mf2 = ['price', 'ln_price', 'THC', 'THC2', 'THC3', 'THC4', 'THC:rating', 
       'body', 'category_name', 'dname', 'license_type', 'listing_name',
       'name', 'prices', 'region', 'zip', 'gram', 'Strain Name', 'MatchedName',
       'Match', 'Matched Strain', '_Index', 'Match Score', 'MatchNew'
       ]
mf3 = ['price', 'ln_price', 'THC', 'THC3', 'THC4', 'THC:rating', 
       'body', 'category_name', 'dname', 'license_type', 'listing_name',
       'name', 'prices', 'region', 'zip', 'gram', 'Strain Name', 'MatchedName',
       'Match', 'Matched Strain', '_Index', 'Match Score', 'MatchNew'
       ]

# Targets
mt1 = ['price']
mt2 = ['ln_price']
mt3 = ['price']

# Set X and y 
features = get_features(mf1, a_df)
target = get_target(mt1, a_df)

design = target + features
analysis_df = a_df[design]

analysis_df = analysis_df.dropna()

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
analysis_df.columns = analysis_df.columns.str.replace(",", "")
analysis_df.columns = analysis_df.columns.str.strip()

# Create formula for patsy
my_formula = analysis_df.columns[0] + " ~ " + \
" + ".join(list(analysis_df.columns[1:len(a_df.columns)])) + ' - 1'

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

## Check all models on training data

grids = score_ht_models(X_train_scale, y_train, scoring = 'r2')

## Run optimized xgboost

cv_params = {'max_depth': [3,5], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.3, 'n_estimators': 1000, 'seed':0, 'subsample': 0.3, 
              'colsample_bytree': 0.3, 'objective': 'linear:regression'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'r2', cv = 5, n_jobs = -1) 
optimized_GBM.fit(X_train_scale, y_train)


xgdmat = xgb.DMatrix(X_train_scale[0:100000], y_train[0:100000]) # Create our DMatrix to make XGBoost more efficient

our_params = {
             "objective": "reg:linear",
             'min_child_weight': 3,
             'subsample': 0.3,
             'gamma': 0.3,
             'colsample_bytree': 0.3,
             'learning_rate': 0.3,
             'max_depth': 5, 
             'reg_alpha': 5
         } 


cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 300, nfold = 3,
                metrics = ['rmse'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error


cv_xgb.head(5)



# It's clear that xgboost is the best performing model
# Now let's check for overfitting
grids['xgb'].best_estimator_.score(X_train_scale, y_train)
grids['xgb'].best_estimator_.score(X_test_scale, y_test)


## Fit the test data with the best model - need to streamline from above


## Predicted vs. Actual

graph_predVSact(X, y, 'xgb', 20, 20, 'Actual $/gram', 'Predicted $/gram')




    




