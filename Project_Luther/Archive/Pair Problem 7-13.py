#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 07:39:50 2018

@author: ajdavis
"""


import random

sims = 1000

doors = ["FML", "FML", "ca$h"]

    
# Switching doors strategy

wins = 0
losses = 0

for i in range(sims):
    random.shuffle(doors) # shuffle door assignment
    d = random.randrange(3)     # contestant picks door d
    sequence = range(3)    # set sequence for door selection in each iteration
    random.shuffle(list(sequence))    # shuffle sequence
    for k in sequence: # host picks door k
        if k == d or doors[k] == "ca$h":
            continue
    if doors[d] == "ca$h":
        losses += 1
    else:
        wins += 1
        
print("The switching strategy has %s wins and %s losses" % (wins, losses))

# Not switching doors strategy

wins = 0
losses = 0

for i in range(sims):
    random.shuffle(doors)
    d = random.randrange(3)
    sequence = range(3)
    random.shuffle(list(sequence))
    for k in sequence:
        if k == d or doors[k] == "ca$h":
            continue
    if doors[d] == "ca$h":
        wins += 1
    else:
        losses += 1
        
print("The non-switching strategy has %s wins and %s losses" % (wins, losses))

# Not switching doors strategy

wins = 0
losses = 0
