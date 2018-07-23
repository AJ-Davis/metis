#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:52:20 2018

@author: ajdavis
"""
#Set up system

import sys
print("Python Version:", sys.version)



# Set up Pandas and other libs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from IPython.display import Image
import urllib
import json
import geocoder

gc_df = pd.read_csv("DOITT_SUBWAY_ENTRANCE_01_13SEPT2010.csv")

# Clean the geocode string
gcs = list(gc_df['the_geom'].str.replace('POINT \(', '').str.replace('\)', '').str.split(' '))
    
# Swap lat and lons
for i in gcs:
    i[0], i[1] = i[1], i[0]

# Convert strings to float
gcs_float = []
for i in gcs:
    new = [float(x) for x in i]
    gcs_float.append(new)
    
def get_zips(list_gcs):
    zips = []
    for gc in list_gcs:
        g = geocoder.google(gc, method = 'reverse')
        zc = g.postal
        zips.append(zc)
    zip_df = pd.DataFrame(
    {'STATIONS': list(gc_df['NAME']),
     'GEOCODE': list_gcs,
     'ZIPS': zips
    })
    return zip_df

 
station_geo = get_zips(gcs)

# match names
url = 'http://web.mta.info/developers/data/nyct/turnstile/turnstile_180630.txt'
turn = pd.read_csv(url, usecols=['STATION'], header = 0)

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

names_array=[]
ratio_array=[]
def match_names(wrong_names,correct_names):
    for row in correct_names:
        x=process.extractOne(row, wrong_names)
        names_array.append(x[0])
        ratio_array.append(x[1])
    return names_array,ratio_array

wrong_names = gc_df['NAME'].dropna().unique()
correct_names = turn['STATION'].dropna().unique()

name_match,ratio_match=match_names(wrong_names,correct_names)

df_match = pd.DataFrame(
        {'TURN': pd.Series(correct_names),
         'MATCH': pd.Series(name_match),
         'RATIO': pd.Series(ratio_match)      
         })

#Merge matched file to zips
merged = pd.merge(df_match, station_geo, 
                  left_on = 'MATCH', 
                  right_on = 'STATIONS', 
                  how = 'left').dropna()
    
# Ancillary Data Sets


demo = pd.read_csv("Demographic_Statistics_By_Zip_Code.csv") # Demographics by zip
demo = demo[['JURISDICTION NAME', 'PERCENT FEMALE']].dropna()

# Merge with station data
merged['ZIPS'] = merged['ZIPS'].astype(str).astype(int)
merged_demo = pd.merge(merged, demo, 
                  left_on = 'ZIPS', 
                  right_on = 'JURISDICTION NAME', 
                  how = 'left')

zillow = pd.read_csv("Zip_MedianValuePerSqft_AllHomes.csv") # Median home prices by zip
zillow = zillow[zillow['City']=='New York']
zillow = zillow[['RegionName', '2018-05']]

merged_demo_zillow = pd.merge(merged_demo, zillow, 
                  left_on = 'ZIPS', 
                  right_on = 'RegionName', 
                  how = 'left')

df = merged_demo_zillow.sort_values(by=['PERCENT FEMALE', '2018-05'], 
                                    ascending = False)
