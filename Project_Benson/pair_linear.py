#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 07:37:49 2018

@author: ajdavis
"""
import numpy as np
x = np.matrix([[5,3,7],[2,4,1]])
w = np.array([100, 10, 1])
y = np.array([537,241])

np.dot(x,w)

# A
X = np.array([[2]])
y = np.array([8])

# B
X = np.array([[0]])
y = np.array([8])

# C

X = np.array([[2, 4]])
y = np.array([8])

# D
X = np.array([[2, 4], [0, 1]])
y = np.array([8, 3])

# E
X = np.array([[2, 4], [0, 1], [9, 5]])
y = np.array([8, 3, 1])

# F
X = np.array([[2, 2], [3, 3]])
y = np.array([4, 6])

# G
X = np.array([['dog'], ['cat']])
y = np.array([8, 6])

# H
X = np.array([[1, 0], [0, 1]])
y = np.array([8, 6])

# I
X = np.array([[1, 1, 0], [1, 0, 1]])
y = np.array([8, 6])

# J
X = np.array([[1, 0], [1, 1]])
y = np.array([8, 6])
