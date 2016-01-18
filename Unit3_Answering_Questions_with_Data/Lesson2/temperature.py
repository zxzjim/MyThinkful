
# coding: utf-8

# In[141]:

get_ipython().magic(u'matplotlib inline')
import requests
import collections
import datetime
import time
from pandas.io.json import json_normalize
import pandas as pd
import sqlite3 as lite
from dateutil.parser import parse
import matplotlib.pyplot as plt


# In[6]:

cities_and_pos = collections.defaultdict()
cities_and_pos = { "Atlanta": '33.762909,-84.422675',
            "Austin": '30.303936,-97.754355',
            "Boston": '42.331960,-71.020173',
            "Chicago": '41.837551,-87.681844',
            "Cleveland": '41.478462,-81.679435'
        }
print cities_and_pos


# In[69]:

'''print datetime.datetime.now()
now = datetime.datetime.now()
now_str = now.strftime('%Y-%m-%dT%H:%M:%S')
month_ago = now - datetime.timedelta(days=30)
month_ago_str = month_ago.strftime('%Y-%m-%dT%H:%M:%S')
print "https://api.forecast.io/forecast/1eb174d457808f327bd58e03b98e5c18/%s" % (cities_and_pos['Chicago'] + ',' + now_str)
print "https://api.forecast.io/forecast/1eb174d457808f327bd58e03b98e5c18/%s" % (cities_and_pos['Chicago'] + ',' + month_ago_str)
r = requests.get("https://api.forecast.io/forecast/1eb174d457808f327bd58e03b98e5c18/%s" % (cities_and_pos['Chicago'] + ',' + now_str))
print r
print r.json()['daily']['data'][0]['temperatureMax']
#print city_weather_df.head()
city_max_temp = r.json()['daily']['data'][0]['temperatureMax']
print city_max_temp'''


# In[109]:

daily_max_temp_dict = collections.defaultdict()
city_max_temp_dict = collections.defaultdict()
for d in range(0,31):
    start_date = datetime.datetime.now() - datetime.timedelta(days=d)
    start_date_str = start_date.strftime('%Y-%m-%dT12:00:00')
    print start_date_str
    for city, pos in cities_and_pos.items():
        print "https://api.forecast.io/forecast/1eb174d457808f327bd58e03b98e5c18/%s" % (cities_and_pos[city] + ',' + start_date_str)
        r = requests.get("https://api.forecast.io/forecast/1eb174d457808f327bd58e03b98e5c18/%s" % (cities_and_pos[city] + ',' + start_date_str))
        if 'daily' in r.json():
            if 'temperatureMax' in r.json()['daily']['data'][0]:
                city_max_temp = r.json()['daily']['data'][0]['temperatureMax']
                print city_max_temp
                city_max_temp_dict[city] = city_max_temp
            else:
                city_max_temp_dict[city] = None
    daily_max_temp_dict[start_date_str] = city_max_temp_dict
    city_max_temp_dict = collections.defaultdict()

        
     


# In[113]:

for t, cd in daily_max_temp_dict.iteritems():
    print t
    for city, temp in cd.iteritems():
        print parse(t).strftime('%s'), city, temp


# In[134]:

conn = lite.connect('weather.db')
cur = conn.cursor()
cur.execute('drop table if exists daily_maxtemps;')
create_sql = 'create table daily_maxtemps (day INT PRIMARY KEY, '+', '.join(x + ' REAL' for x in cities_and_pos.iterkeys())+')'
cur.execute(create_sql)

for t, ct in daily_max_temp_dict.iteritems():
    insert_sql = 'INSERT INTO daily_maxtemps(day) VALUES(%s)' % (parse(t).strftime('%s'))
    print insert_sql
    cur.execute(insert_sql)
    for city, temp in ct.iteritems():
        update_sql = 'UPDATE daily_maxtemps SET '+ city +' = ' + str(temp) + ' WHERE day = ' + parse(t).strftime('%s')
        print update_sql
        cur.execute('UPDATE daily_maxtemps SET '+ city +' = ' + str(temp) + ' WHERE day = ' + parse(t).strftime('%s'))


# In[153]:

df = pd.read_sql('select * from daily_maxtemps', conn)
print df.head()
print df.describe()
print df.columns[1::]


# In[160]:

for city in df.columns:
    plt.bar(df.day, df[city])
    plt.show()

