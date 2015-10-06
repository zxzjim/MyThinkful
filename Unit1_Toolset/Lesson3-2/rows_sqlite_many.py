import sqlite3 as lite
con = lite.connect('getting_started.db')
cities = (('Bremen', 'Bremen'),('Hannover', 'Lower Saxony'))
weather = (('Las Vegas',2013,'July','December',78),('Atlanta',2013,'July','January',70))
with con:
    cur = con.cursor()
    cur.executemany("INSERT INTO cities VALUES(?,?)", cities)
    cur.executemany("INSERT INTO weather VALUES(?,?,?,?,?)", weather)