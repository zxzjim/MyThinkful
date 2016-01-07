
# coding: utf-8

# In[87]:

import pandas as pd
import sqlite3 as lite
import collections
import time
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt


con = lite.connect('citi_bike.db')
cur = con.cursor()

df = pd.read_sql("SELECT * FROM available_bikes ORDER BY execution_time", con, index_col='execution_time')
df.head()


# In[88]:

station_change = collections.defaultdict(int)
for col in df.columns:
    change = 0
    station_series = df[col].tolist()
    for k, v in enumerate(station_series):
        if k < len(station_series) - 1:
            change += abs(station_series[k] - station_series[k+1])
    station_change[int(col[1:])] = change    


# In[89]:

for k, v in station_change.items():
    print k, v


# In[90]:

max_station = max(station_change, key=station_change.get)
print max_station
print station_change[max_station]


# In[91]:

cur.execute('SELECT id, stationname, latitude, longitude FROM citibike_reference WHERE id = ?', (max_station,))
data = cur.fetchone()
print("The most active station is station %s, at %s, latitude: %s, longitude: %s, " % data)
print("With %d bicycles coming and going in the hour between %s and %s" % (
    station_change[max_station],
    datetime.datetime.fromtimestamp(int(df.index[0])).strftime('%Y-%m-%dT%H:%M:%S'),
    datetime.datetime.fromtimestamp(int(df.index[-1])).strftime('%Y-%m-%dT%H:%M:%S')
    ))


# In[92]:

plt.bar(station_change.keys(), station_change.values())
plt.show()

