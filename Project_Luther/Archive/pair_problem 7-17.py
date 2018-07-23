#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:26:11 2018

@author: ajdavis
"""

from generate_sample import get_sample_success

sims = 10000

def sim_phones(sims, p0, p1, count):
    comp_results = []
    for i in range(sims):
        # True is the case when we get the wrong result i.e. p0 fewer defects than p1
        result = get_sample_success(p0, count) < get_sample_success(p1, count)
        comp_results.append(result)
    return sum(comp_results)

sim_phones(sims, 0.05, 0.03, 1000) #1
sim_phones(sims, 0.05, 0.04, 1000) #2
sim_phones(sims, 0.05, 0.04, 20000) #3
sim_phones(sims, 0.05, 0.048, 135000) #4
    






