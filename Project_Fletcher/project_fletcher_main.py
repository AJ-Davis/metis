#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 20:16:44 2018

@author: ajdavis
"""

## Get Data
# Load Leafly and Washington Lab testing data
effects = pd.read_csv('/Users/ajdavis/github/metis/Project_Fletcher/Data/cannabis.csv')
content = pd.read_csv('/Users/ajdavis/github/metis/Project_Fletcher/Data/cannabinoid_content.csv', encoding='iso-8859-1')

mask = (content['InventoryLabel'] == 'Flower Lot')
cols = ['TestStrain', 'THCmax', 'CBDmax']

content = content[mask]
content = content[cols]

content = content.groupby(['TestStrain']).mean().reset_index()

content['MatchStrain'] = content['TestStrain'].str.strip().str.replace(' ', '').str.replace('.', '').str.replace("'", '').str.replace('/', '').str.replace('-', '').str.replace('#', '').str.lower()
effects['MatchStrain'] = effects['Strain'].str.strip().str.replace(' ', '').str.replace('.', '').str.replace("'", '').str.replace('/', '').str.replace('-', '').str.replace('#', '').str.lower()

# Fuzzy match leafly and lab testing data
# Columns to match on from df_left
left_on = ['MatchStrain']

# Columns to match on from df_right
right_on = ['MatchStrain']

# The link table potentially contains several matches for each record
# test = fuzzymatcher.link_table(effects, content, left_on, right_on)
df = fuzzymatcher.fuzzy_left_join(effects, content, left_on, right_on)
df = df.dropna().reset_index()

wl_df = pd.read_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/wl_df.pkl')
wl_df['thc_avg_strain'] = ((pd.to_numeric(wl_df['thc_avg_strain'])/100)*30).fillna(0)
wl_df['thc_avg_type'] = ((pd.to_numeric(wl_df['thc_avg_type'])/100)*30).fillna(0)
wl_df['thc_high_strain'] = ((pd.to_numeric(wl_df['thc_high_strain'])/100)*30).fillna(0)

wl_df['cbd_avg_strain'] = ((pd.to_numeric(wl_df['cbd_avg_strain'])/100)*30).fillna(0)
wl_df['cbd_avg_type'] = ((pd.to_numeric(wl_df['cbd_avg_type'])/100)*30).fillna(0)
wl_df['cbd_high_strain'] = ((pd.to_numeric(wl_df['cbd_high_strain'])/100)*30).fillna(0)


wl_df_effects = wl_df.iloc[:, 8:].apply(lambda x: pd.to_numeric(x)).fillna(0)
wl_df_effects = wl_df_effects.apply(lambda x: x/100)
wl_df = pd.concat([wl_df.iloc[:, :8], wl_df_effects], axis = 1)


# Create sentiment scores
df['Sentiment'] = get_sentiment(df['Description'])


# Get WikiLeaf data

wl_links = get_wllinks()
wl_df = get_wldata(wl_links)

wl_df.to_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/wl_df.pkl')

# Get Cannabis Reports Data

search = df['MatchStrain_left'].tolist()
cr_links = list(itertools.chain.from_iterable(list(map(get_crlinks, search))))
ucpcs = [d['ucpc'] for d in cr_links]

cr_effects = list(itertools.chain.from_iterable(list(map(get_creffects, ucpcs))))





## Prep Leafly data for analysis

columns = ['TestStrain', 'THCmax', 'CBDmax', 'Rating', 'Effects', 'Flavor' , 'Type', 'Sentiment']

df_rec = df[columns]
df_rec = df_rec.dropna() # Remove na's

df_effects = df_rec['Effects'].str.get_dummies(sep=',')
df_rec = pd.concat([df_rec, df_effects], axis=1)
df_flavors = df_rec['Flavor'].str.get_dummies(sep=',')
df_rec = pd.concat([df_rec, df_flavors], axis=1)

df_rec = pd.get_dummies(df_rec, columns=['Type'], drop_first = True) # Create dummies
df_rec.drop(['Effects', 'Flavor'], axis = 1, inplace = True)

df_rec.set_index('TestStrain', inplace=True)

df_rec.to_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df_leaf.pkl')
wl_df.to_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df_wl.pkl')

## Recommender


ss = StandardScaler()
transformed_data = ss.fit_transform(df_rec)
df_rec_ss = pd.DataFrame(transformed_data, columns=df_rec.columns,
                      index = df_rec.index)

dists = cosine_similarity(df_rec_ss)
dists = pd.DataFrame(dists, columns=df_rec.index)
dists.index = dists.columns


# Pickle for flask app
df_rec.to_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df_rec.pkl')
df.to_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df.pkl')

## Clustering

np.random.seed(0)

df_rec = pd.read_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df_leaf.pkl')
wl_df = pd.read_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df_wl.pkl')

wl_df['CBDmax'] = wl_df['cbd_high_strain']
wl_df['THCmax'] = wl_df['thc_high_strain']
wl_df['Hungry'] = wl_df['use_hungry']
wl_df['Relaxed'] = wl_df['use_relax']
wl_df['Sleepy'] = wl_df['use_sleep']
wl_df['Talkative'] = wl_df['use_social']
wl_df['Creative'] = wl_df['use_creativity']
wl_df['Energetic'] = wl_df['use_energy']
wl_df['Focused'] = wl_df['use_focus']
wl_df['Happy'] = wl_df['use_happy']

mask = ['CBDmax', 'THCmax', 'Hungry', 'Relaxed', 'Sleepy', 'Talkative',
        'Creative', 'Energetic', 'Focused', 'Happy']

df_wl = wl_df[mask]
df_leaf = df_rec[mask]

leaf_tab = do_clustering(df_leaf)
wl_tab = do_clustering(df_wl)


## Export csv for Tableau
leaf_tab.to_csv('/Users/ajdavis/github/metis/Project_Fletcher/Data/leaf_tab.csv')
wl_tab.to_csv('/Users/ajdavis/github/metis/Project_Fletcher/Data/wl_tab.csv')

## Analysis of clusters
leaf_means = get_chemovar_means(leaf_tab, clustering_algorithms)
wl_mean = get_chemovar_means(wl_tab, clustering_algorithms)

# Just look at most prominant medicinal effects
effects = ['Hungry', 'Relaxed', 'Sleepy', 'Talkative',
        'Creative', 'Energetic', 'Focused', 'Happy']

leaf_effects_df = pd.concat(get_pvals(leaf_tab, effects, clustering_algorithms))
wl_effects_df = pd.concat(get_pvals(wl_tab, effects, clustering_algorithms))

leaf_effects_df.to_csv('/Users/ajdavis/github/metis/Project_Fletcher/Data/leaf_effects_df.csv')
wl_effects_df.to_csv('/Users/ajdavis/github/metis/Project_Fletcher/Data/wl_effects_df.csv')

