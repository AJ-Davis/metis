#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 07:37:31 2018

@author: ajdavis
"""

def check_parentheses(par):
    count = 0
    for i in par:
        if i == '(':
            count += 1
        elif i == ')':
            count -= 1
        if count < 0:
            return False
    return count == 0