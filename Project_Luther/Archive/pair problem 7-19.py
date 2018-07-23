#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:05:15 2018

@author: ajdavis
"""

inputs = ['Ada Lovelace', 'Haskell Curry', 'AJ Davis']
from nltk.corpus import words
nltk.download()
word_list = words.words()

def dank_username(name):
    name = name.lower()
    username = []
    
    for w in word_list:
        w = w.lower()
        if set(w) - set(name):
            continue
    for i in range(len(w)):
        first, second = w[:i], w[i:]
        if first in name and second in name:
            username.append(w)
            break
    max_length = len(max(username, key = len))
    return [w for w in username if len(w) == max_length]