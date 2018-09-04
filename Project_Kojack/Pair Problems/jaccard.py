#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 07:46:51 2018

@author: ajdavis
"""

# Jaccard Similarity = (the number in both sets) / (the number in either set) * 100
# J(X,Y) = 1 - (|X∩Y| / |X∪Y|)

l1 = [0, 5, 3, 5, 'cat']
l2 = [9, 6, 5, 8, 6]



def j(l1, l2):
    if l1 == [] and l2 == []:
        return 1
    else:
        l1 = set(l1)
        l2 = set(l2)
        inter = len(list(set(l1).intersection(l2)))
        union = (len(l1) + len(l2)) - inter
        return (float(inter / union))

j(l1, l2)
