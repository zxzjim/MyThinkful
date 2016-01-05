# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:37:40 2016

@author: zhangxinzhou
"""

import requests
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3 as lite
import time
from dateutil.parser import parse 
import collections

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

test_stations = df[df['testStation'] == True]
print 'number of stations in test: ' + str(len(test_stations))

inService_stations = df[df['statusValue'] == 'In Service']
print 'number of stations in service: ' + str(len(inService_stations))

df['occupiedDocks'] = df.totalDocks - df.availableDocks
print df.occupiedDocks.mean()
print df.occupiedDocks.median()

df2 = inService_stations
df2['occupiedDocks'] = df.totalDocks - df.availableDocks
print df2.occupiedDocks.mean()
print df2.occupiedDocks.median()


#saving the data to the database

con = lite.connect('citi_bike.db')
cur = con.cursor()

with con:
    cur.execute('DROP TABLE IF EXISTS citibike_reference')
    cur.execute('CREATE TABLE citibike_reference (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, landMark TEXT, latitude NUMERIC, location TEXT )')
sql = 'INSERT INTO citibike_reference (id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)'
with con:
    for station in r['stationBeanList']:
        cur.execute(sql,(station['id'],station['totalDocks'],station['city'],station['altitude'],station['stAddress2'],station['longitude'],station['postalCode'],station['testStation'],station['stAddress1'],station['stationName'],station['landMark'],station['latitude'],station['location']))

station_ids = ['_' + str(x) + ' INT' for x in df.id]
with con:
    cur.execute('CREATE TABLE IF NOT EXISTS available_bikes (execution_time INT, '+', '.join(station_ids)+');')

exec_time = parse(r['executionTime'])
with con:
    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time.strftime('%s'),))

id_bikes = collections.defaultdict(int)
for station in r['stationBeanList']:
    id_bikes[station['id']] = station['availableBikes']
with con:
    for k, v in id_bikes.iteritems():
        cur.execute('UPDATE available_bikes SET _'+ str(k) + '=' + str(v) + ' WHERE execution_time ='+ exec_time.strftime('%s') +';')
