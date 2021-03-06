#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:38:46 2018

@author: ajdavis
"""


## Read in data 
df = pd.read_stata('SAMHDA.dta')


# Select relevant columns based on review of the survey codebook
feature_space = ['pnrnmlif', 'mrdaypyr', 'mjever', 'mjrec', 'mjyrtot', 'service', 'health', 
           'sexatract','sexident', 'AGE2', 'irsex', 'irmaritstat', 
           'IREDUHIGHST2', 'eduenroll','CATAG6', 'NEWRACE2', 'WRKSTATWK2',
           'WRKDHRSWK2', 'wrksickmo', 'wrkskipmo', 'wrktstdrg',
           'wrktsthir', 'wrktstrdm', 'WRKTST1ST', 'wrkokpreh',
           'wrkokrand', 'IRKI17_2', 'hltinmnt', 'hlcnotyr', 
           'irfstamp', 'IRPINC3', 'COUTYP2']

df = df[feature_space] #subset df


## Munge data
df_munge = munge_df(df) # Clean data and feature engineer


## Prep data for analysis

# Set target, then non-dummies, then dummies
feats_to_try = ['target', 'service', 'hlcnotyr',
                'health', 'COUTYP2', 'irsex', 'irmaritstat', 'IREDUHIGHST2', 
                'eduenroll', 'CATAG6', 'NEWRACE2', 'irfstamp', 'IRPINC3']

#Index argument for where to start dummies e.g. at the third column
X, y = analysis_prep(df_munge, feats_to_try, 3) 

# Balance the data
X_smoted, y_smoted = SMOTE(random_state=42).fit_sample(X,y.values.ravel())
X_smoted = pd.DataFrame(X_smoted, columns = X.columns) # add column names back

# Train/test split
X_smoted_train, X_smoted_test, y_smoted_train, y_smoted_test = train_test_split(X_smoted,y_smoted)



## Feature Selection
# Index argument is for how many top features to return per method
feats = feature_selection(X_smoted_train, y_smoted_train, 20)

# After reviewing the selected features stack on one column
# to identify which features were identified most often
a = feats.iloc[:,0]
b = feats.iloc[:,1]
c = feats.iloc[:,2]
d = feats.iloc[:,3]

top_feats1 = pd.concat([a, b, c, d], ignore_index=True).value_counts().head(20)

feats_to_keep = top_feats.index.values

# Subset data to only include top features
X_smoted_train_select = X_smoted_train[feats_to_keep]
X_smoted_test_select = X_smoted_test[feats_to_keep]
    

## Score models
# Given business case, we are most interested in identifying positive cases,
# i.e. the type of cannabis consumers who dispensary users will benefit most
# from marketing to. More precisely, I want to find all the data points of interest
# in the dataset. We can accept low precision because the cost of marketing to
# false postive likely won't be high and may have benefit.


# Scale the data
ssX = StandardScaler()
X_smoted_train_select_scaled = ssX.fit_transform(X_smoted_train_select)
X_smoted_test_select_scaled = ssX.transform(X_smoted_test_select)

# Loop through all relevant models, tune, and score
grids = score_ht_models(X_smoted_train_select_scaled, y_smoted_train, scoring = 'recall')

# Score models that I am not hyperparameter tuning
# Also check for impact over baseline (Dummy Classifier)
scores_other = score_models(X_smoted_train_select_scaled, y_smoted_train, scoring = 'recall')



# Score best model on test data to check for overfitting
grids['tree'].best_estimator_.score(X_smoted_test_select_scaled, y_smoted_test)
grids['forest'].best_estimator_.score(X_smoted_test_select_scaled, y_smoted_test)
grids['svc'].best_estimator_.score(X_smoted_test_select_scaled, y_smoted_test)

# Check confusion matrix and classification report for best model
y_pred_test = grids['forest'].best_estimator_.predict(X_smoted_test_select_scaled)
y_pred_train = grids['forest'].best_estimator_.predict(X_smoted_train_select_scaled)

conf_mat_test = confusion_matrix(y_true=y_smoted_test, y_pred=y_pred_test)
conf_mat_train = confusion_matrix(y_true=y_smoted_train, y_pred=y_pred_train)

cm_test = print_confusion_matrix(conf_mat_test, ['Class 0', 'Class 1'])
cm_train = print_confusion_matrix(conf_mat_train, ['Class 0', 'Class 1'])

cr_test = classification_report(y_smoted_test,y_pred_test)
cr_train = classification_report(y_smoted_train,y_pred_train)


## Prepare to ship

# Create pipeline
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('forest', RandomForestClassifier(**grids['forest'].best_params_))])

# Subset full feature set with top features
X_smoted_select = X_smoted[feats_to_keep]

# Rename columns something interaptible for web app
X_smoted_select.columns = ['Health',  'Enrolled', 'Ethnicity1', 'Income', 'Education1',
                           'Marital_Status1', 'Age', 'Ethnicity2', 'Marital_Status2',
                           'Education2']

# Fit best model
pipeline.fit(X_smoted_select,y_smoted)

# Save model
pickle.dump(pipeline, open('/Users/ajdavis/github/metis/Project_McNulty/Web App/model.pkl', 'wb'))


## Additional feature transform of best model
## Hyperparameter tuning of best model
## Output results
## Tables
## Graphs

