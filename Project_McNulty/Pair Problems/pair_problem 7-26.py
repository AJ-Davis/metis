#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 07:51:21 2018

@author: ajdavis
"""

from math import factorial as fac
import random
import numpy as np

def ways(n, d):
    return fac(n)/(fac(d)*fac(n-d))

def ways2(n, d):
    return fac(n)/(fac(n-d) * fac(d-1))

ways(5, 4)
ways2(5, 2)

d = {1:f, -1:b, 0:s}
   
nl = []
nnl = []
def list_gen(l):
    for i in range(20):
        n = random.choice(d)
        nl.append(n)
    if np.sum(nl) == 2:
        nnl.append(nl)
    return nl

list_gen(l)       
        
    
    
    
    
