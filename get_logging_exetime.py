#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 20:58:38 2025

@author: marcop
"""

from datetime import datetime #, timedelta
import sys

if len(sys.argv) > 1:
    f = sys.argv[1]
else:
    raise ValueError("no file argument to read")

tformat = '%Y-%m-%d %H:%M:%S.s%f'

def time_from_line(line : str) -> datetime:
    t0 = ' '.join(line.split(' ')[:2])
    return datetime.strptime(t0, tformat)

with open(f, 'r') as fin:
    a = fin.readlines()
    
    d0 = time_from_line(a[0])
    du = time_from_line(a[-1])
    
    exetime = (du - d0).total_seconds()
    
    print(exetime)
    
# with open('/home/marcop/Documents/work/corberan/logs/amelia_mi_70_epsg4326_parallel.log', 'r') as fin:
#     a = fin.readlines()
#     t0 = ' '.join(a[0].split(' ')[:2])
#     tu = ' '.join(a[-1].split(' ')[:2])
    
#     d0 = datetime.strptime(t0, tformat)
#     du = datetime.strptime(tu, tformat)
    
#     exetime_par = (du - d0).total_seconds()
    
    