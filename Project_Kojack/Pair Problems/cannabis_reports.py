#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:33:08 2018

@author: ajdavis
"""

from cannabis_reports import CannabisReports
cr = CannabisReports('API_KEY')
cr.__apis__

strains = []
for i in range(25):
    for strain in cr.Strains.list():
        s = strain.serialize()
        strains.append(s)

strains[0]['ucpc']


info = []
for strain in strains:
    s = cr.Strains.get(strain['ucpc'])
    ss = s.serialize()
    info.append(ss)