# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:14:40 2016

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

conn = lite.connect('citi_bike.db')
cur = conn.cursor()

print 'dropping reference table, creating reference table'
with conn:
    cur.execute('DROP TABLE IF EXISTS citibike_reference')
    cur.execute('CREATE TABLE citibike_reference (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, landMark TEXT, latitude NUMERIC, location TEXT )')
    
citibike_ref_sql = 'INSERT INTO citibike_reference (id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)'

for i in range(60):
    
    print 'getting data from api'
    r = requests.get('http://www.citibikenyc.com/stations/json')
    df = json_normalize(r.json()['stationBeanList'])
    station_ids = ['_' + str(x) + ' INT' for x in df.id]
    exec_time = parse(r.json()['executionTime']).strftime('%s')
    print 'execution time is: ' + r.json()['executionTime']
    
    print 'filling up ref table'
    cur.execute('DELETE FROM citibike_reference')
    for station in r.json()['stationBeanList']:
        cur.execute(citibike_ref_sql,(station['id'],station['totalDocks'],station['city'],station['altitude'],station['stAddress2'],station['longitude'],station['postalCode'],station['testStation'],station['stAddress1'],station['stationName'],station['landMark'],station['latitude'],station['location']))
            
    
    cur.execute('CREATE TABLE IF NOT EXISTS available_bikes (execution_time INT, '+', '.join(station_ids)+')')
    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time,))
    
    print 'filling up available bikes...'
    for station in r.json()['stationBeanList']:
        cur.execute('UPDATE available_bikes SET _%d = %d WHERE execution_time = %s' % (station['id'], station['availableBikes'], exec_time))
    
    conn.commit()
    
    time.sleep(60)

conn.close()