#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 07:53:45 2018

@author: ajdavis
"""

def ways(C, m, n):
    table = [0 for k in range(n + 1)]
    
    table[0]= 1
    
    for i in range(0, m):
        for j in range(C[i], n+1):
            table[j] += table[j - C[i]]
    return table[n]

C = [1, 5, 10, 25]
m = len(C)
n = 100

ways(C, m, n)
