# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:37:40 2016

@author: zhangxinzhou
"""

import requests
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import pandas as pd

r = requests.get('http://www.citibikenyc.com/stations/json')
r = r.json()
print len(r['stationBeanList'])

key_list = []
for station in r['stationBeanList']:
    for k in station.keys():
        if k not in key_list:
            key_list.append(k)
            
df = json_normalize(r['stationBeanList'])
df['availableBikes'].hist()
plt.show()

df['totalDocks'].hist()
plt.show()

df['availableDocks'].hist()
plt.show()

test_stations = []
for station in r['stationBeanList']:
    if station['testStation'] == True:
        test_stations.append(station)
print 'number of stations in test: ' + str(len(test_stations))

inService_stations = []
for station in r['stationBeanList']:
    if station['statusValue'] == 'In Service':
        inService_stations.append(station)
print 'number of stations in service: ' + str(len(inService_stations))

df['occupiedDocks'] = df.totalDocks - df.availableDocks
print df.occupiedDocks.mean()
print df.occupiedDocks.median()

df2 = json_normalize(inService_stations)
df2['occupiedDocks'] = df.totalDocks - df.availableDocks
print df2.occupiedDocks.mean()
print df2.occupiedDocks.median()