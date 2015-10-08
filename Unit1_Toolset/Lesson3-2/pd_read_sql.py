import sqlite3 as lite
import panda as pd
import pandas as pd
con = lite.connect('getting_started.db')
query = "SELECT * FROM cities"
df = pd.read_sql(query, con)