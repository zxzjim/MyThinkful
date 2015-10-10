import sqlite3 as lite
import pandas as pd

con = lite.connect('lesson3-3_challenge.db')

tables = ('cities', 'weather')

cities = (('Washington', 'DC'), 
	('New York City', 'NY'),
	('Boston', 'MA'),
    ('Chicago', 'IL'),
    ('Miami', 'FL'),
    ('Dallas', 'TX'),
    ('Seattle', 'WA'),
    ('Portland', 'OR'),
    ('San Francisco', 'CA'),
    ('Los Angeles', 'CA'),
    ('Houston', 'TX'),
    ('Las Vegas', 'NV'),
    ('Atlanta', 'GA'))

weather = (('New York City', 2013, 'July', 'January', 62),
	('Boston', 2013, 'July', 'January', 59),
	('Chicago', 2013, 'July', 'January', 59),
	('Miami', 2013, 'August', 'January', 84),
	('Dallas', 2013, 'July', 'January', 77),
	('Seattle', 2013, 'July', 'January', 61),
	('Portland', 2013, 'July', 'December', 63),
	('San Francisco', 2013, 'September', 'December', 64),
	('Los Angeles', 2013, 'September', 'December', 75))

with con:
	cur = con.cursor()
	cur.execute("DROP TABLE IF EXISTS cities")
	cur.execute("DROP TABLE IF EXISTS weather")
	cur.execute("CREATE TABLE cities (name text, state text)")
	cur.execute("CREATE TABLE weather (city text, year integer, warm_month text, cold_month text, average_high integer)")
	
	#insert data into the two tables
	cur.executemany("INSERT INTO cities VALUES(?,?)", cities)
	cur.executemany("INSERT INTO weather VALUES(?,?,?,?,?)", weather)

	#join the data together
	query = "SELECT name, state, year, warm_month, cold_month, average_high FROM cities LEFT OUTER JOIN weather ON name = city WHERE warm_month = 'July'"

	#load into Pandas Dataframe
	df = pd.read_sql(query, con)
	
	#print the setence
	output = 'The cities that are warmest in July are: '
	data = zip(df['name'], df['state'])
	for name, state in data:
		output = output + name + ', ' + state + ', '
	print output

