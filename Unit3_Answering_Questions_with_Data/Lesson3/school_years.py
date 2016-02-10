
# coding: utf-8

# In[144]:

from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import DataFrame
import sqlite3 as lite


# In[3]:

url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"
r = requests.get(url)


# In[ ]:

soup = BeautifulSoup(r.content)


# In[11]:

print soup.prettify


# In[8]:

for row in soup('table'):
    print row


# In[138]:

rows_count = len(soup('table')[6].findAll('tr')[7:-1])
for row in soup('table')[6].findAll('tr')[7:rows_count-3]:
    print row



# In[139]:

for i, row in enumerate(soup('table')[6].findAll('tr')[8:rows_count-3]):
    print i, row.find_all('td')[0].string, row.find_all('td')[1].string, row.find_all('td')[4].string, row.find_all('td')[7].string, row.find_all('td')[10].string


# In[127]:

col = []
for c in soup('table')[6].findAll('tr')[7]:
    if c.string is not None:
        #print c.string
        col.append(c.string.strip())
col = filter(None, col)
df = DataFrame(columns= col)
df.dropna
print df.columns


# In[162]:

for i, row in enumerate(soup('table')[6].findAll('tr')[8:rows_count-3]):
    if row.find_all('td')[0].string:
        print i, row.find_all('td')[0].string, row.find_all('td')[1].string, row.find_all('td')[4].string, row.find_all('td')[7].string, row.find_all('td')[10].string
        df.loc[i] = row.find_all('td')[0].string, int(row.find_all('td')[1].string), int(row.find_all('td')[4].string), int(row.find_all('td')[7].string), int(row.find_all('td')[10].string)


# In[163]:

df


# In[151]:

conn = lite.connect('UN_education.db')
cur = conn.cursor()


# In[164]:

cur.execute('DROP TABLE IF EXISTS school_years')
df.to_sql('school_years', conn)


# In[156]:

cur.execute('SELECT * FROM school_years')
for r in cur:
    print r


# In[169]:

print df.describe()


