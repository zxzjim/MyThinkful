import sqlite3 as lite
con = lite.connect('getting_started.db')
with con:
    cur = con.cursor()
    cur.execute("INSERT INTO cities VALUES('Hamburg', 'Hamburg')")
    cur.execute("INSERT INTO cities VALUES('Frankfurt', 'Hessen')")
    cur.execute("INSERT INTO cities VALUES('Dortmund', 'NRW')")
    cur.execute("INSERT INTO weather VALUES('Washington', 2013, 'July', 'January', 67)")
    cur.execute("INSERT INTO weather VALUES('Houston', 2013, 'July', 'January', 66)")