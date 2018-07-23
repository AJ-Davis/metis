#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:35:43 2018

@author: ajdavis
"""

# Set up Pandas and other libs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from IPython.display import Image
import urllib
import json
import geocoder
from sklearn import preprocessing

## Get geo data for mta data stations ##
def get_zips(list_names):
    zips = []
    gcs = []
    for name in list_names:
        g = geocoder.google(name)
        zc = g.postal
        gc = g.latlng
        zips.append(zc)
        gcs.append(gc)
    zip_df = pd.DataFrame(
    {'STATIONS_LOOKUP': stations,
     'STATIONS_MTA': list(turn['STATION'].unique()),
     'ZIPS': zips,
     'GEOCODES': gcs
    })
    return zip_df

url = 'http://web.mta.info/developers/data/nyct/turnstile/turnstile_180630.txt'
turn = pd.read_csv(url, usecols=['STATION'], header = 0)

stations = list(turn['STATION'].unique())

stations = [i + ' station, New York, NY' for i in stations]
geo = get_zips(stations)
geo.to_csv("geo.csv")


missing_zips = {'59 ST':10022, '5 AV/59 ST':10022, '49 ST':10019, 
                '34 ST-HERALD SQ':10001, '28 ST':10016, 
                '23 ST':10010, '14 ST-UNION SQ':10003,
                'BROAD ST':10005, 'CHURCH AV':11226, 'PROSPECT AV':10459, 
                '25 AV':11232, 'CONEY IS-STILLW':11222, 'GRAND ST':11211,
                'ATLANTIC AV':11217, 'SUTTER AV':11207, 'NEW LOTS':11208,
                'HOWARD BCH JFK':11414, 'CENTRAL AV':11221, 
                'SENECA AVE':11385,
                'FOREST AVE':11385, 'FRESH POND RD':11385, 
                '135 ST':10030, '81 ST-MUSEUM':10024, '59 ST COLUMBUS':10023, 
                '14 ST':10011,
                'HIGH ST':11201, 'HOYT-SCHER':11201, 'BROADWAY JCT':11233,
                'JKSN HT-ROOSVLT':11372, 'ELMHURST AV':11373, 
                '67 AV':11375, 
                'FOREST HILLS 71':11375, 'KEW GARDENS':11415, 
                'YORK ST':11201, '4 AV-9 ST':11215, 'DITMAS AV':11218, 
                'NEPTUNE AV':11224,
                'JAMAICA CENTER':11433, 
                'JACKSON AV':10455,
                '233 ST':10466, 'NEREID AV':10466, 'WAKEFIELD/241':10470, 
                'CYPRESS AV': 10454,
                'LONGWOOD AV':10459, '82 ST-JACKSON H':11372, 
                'SUTTER AV-RUTLD':11212, 
                'NEW LOTS AV':11208,
                'FLATBUSH AV-B.C':11210}

geo['ZIPS'] = geo['ZIPS'].fillna(geo['STATIONS_MTA'].map(missing_zips))
geo = geo.dropna()

# =============================================================================
# ## For missing try to  fuzzy match to geodcoded data set ##
# # Import geocoded data set
# gc_df = pd.read_csv("DOITT_SUBWAY_ENTRANCE_01_13SEPT2010.csv")
# 
# # Clean the geocode string
# gcs = list(gc_df['the_geom'].str.replace('POINT \(', '').str.replace('\)', '').str.split(' '))
#     
# # Swap lat and lons
# for i in gcs:
#     i[0], i[1] = i[1], i[0]
# 
# # Convert strings to float
# gcs_float = []
# for i in gcs:
#     new = [float(x) for x in i]
#     gcs_float.append(new)
#     
# def get_zips(list_gcs):
#     zips = []
#     for gc in list_gcs:
#         g = geocoder.google(gc, method = 'reverse')
#         zc = g.postal
#         zips.append(zc)
#     zip_df = pd.DataFrame(
#     {'STATIONS': list(gc_df['NAME']),
#      'GEOCODE': list_gcs,
#      'ZIPS': zips
#     })
#     return zip_df
# 
# gc_geo = get_zips(gcs)
# 
# 
# # Try to match missing stations using fuzzy match
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# 
# names_array=[]
# ratio_array=[]
# def match_names(wrong_names,correct_names):
#     for row in correct_names:
#         x=process.extractOne(row, wrong_names)
#         names_array.append(x[0])
#         ratio_array.append(x[1])
#     return names_array,ratio_array
# 
# wrong_names = list(gc_df['NAME'].dropna().unique())
# correct_names = list(station_geo[station_geo['ZIPS'].isnull()][['STATIONS_MTA']])
# 
# name_match,ratio_match=match_names(wrong_names,correct_names)
# 
# df_match = pd.DataFrame(
#         {'TURN': pd.Series(correct_names),
#          'MATCH': pd.Series(name_match),
#          'RATIO': pd.Series(ratio_match)      
#          })
# 
# #Merge matched file to zips
# merged = pd.merge(df_match, station_geo, 
#                   left_on = 'MATCH', 
#                   right_on = 'STATIONS', 
#                   how = 'left').dropna()
#     
# 
# =============================================================================

## Ancillary Data Sets ##

## Demographic data
demo = pd.read_csv("Demographic_Statistics_By_Zip_Code.csv") # Demographics by zip
demo = demo[['JURISDICTION NAME', 'PERCENT FEMALE']].dropna()


# Merge with station data
geo['ZIPS'] = geo['ZIPS'].astype(str).astype(float)
geo_demo = pd.merge(geo, demo, 
                  left_on = 'ZIPS', 
                  right_on = 'JURISDICTION NAME', 
                  how = 'left')

## Median house price data (proxy for income)
zillow = pd.read_csv("Zip_MedianValuePerSqft_AllHomes.csv") # Median home prices by zip
zillow = zillow[zillow['City']=='New York']
zillow = zillow[['RegionName', '2018-05']]

geo_demo_zillow = pd.merge(geo_demo, zillow, 
                  left_on = 'ZIPS', 
                  right_on = 'RegionName', 
                  how = 'left')


## Tech data
# 'Tech Hotspots' from:
# http://abny.org/images/downloads/2016_nyc_tech_ecosystem_10.17.2017_final_.pdf
# https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm
tech_zips = pd.DataFrame(
        {'ZIPS':[10004, 10005, 10006, 10007, 10038, 10280, # Lower Manhatten
                 11368, 11369, 11370, 11372, 11373, 11377, 11378, # West Queens
                 	10001, 10011, 10018, 10019, 10020, 10036, # Midtown South
                     	11206, 11221, 11237, 	11211, 11222] # Brooklyn Tech Triangle
                })
    
geo_demo_zillow['Tech'] = geo_demo_zillow['ZIPS'].isin(tech_zips['ZIPS']).astype(int)

## Turnstile data
td = pd.read_csv('output.csv')
geo_demo_zillow_td = pd.merge(td, geo_demo_zillow, 
                              left_on = 'STATION',
                              right_on = 'STATIONS_MTA',
                              how = 'left')
    
# =============================================================================
# ## Religious data
# rel = pd.read_stata('U.S. Religion Census Religious Congregations and Membership Study, 2010 (Metro Area File).dta')
#     
# =============================================================================

## Scoring algorithm
#Filter scoring features for normalization
geo_demo_zillow_td_na = geo_demo_zillow_td.dropna()
features = geo_demo_zillow_td_na[['FTRAFFIC', 'PERCENT FEMALE', '2018-05', 'Tech']]
ids = geo_demo_zillow_td_na[['STATION', 'DOW']]
# normalize values
x = features.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=features.columns)


# create weights and score
weights = np.array([7,1,1,1])
df['SCORE'] = (df * weights).sum(axis=1)
df_final = pd.concat([ids.reset_index(drop=True), df], axis=1)

# Top Ten with ancillary vars
df_final = df_final.sort_values('SCORE', ascending = False).head(10)

df_final.rename(columns = {'2018-05':'HOME PRICE',
                           'Tech': 'TECH HUB'}, inplace = True)
df_final['TECH HUB'].replace(to_replace = 1, value = 'YES', inplace = True)

# Tope Ten just foot traffic
df_final_td = td.sort_values('FTRAFFIC', ascending = False).head(10)




