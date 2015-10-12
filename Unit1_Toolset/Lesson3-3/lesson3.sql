SELECT state, AVG(average_high) FROM cities
LEFT OUTER JOIN weather 
ON name = city
GROUP BY state

SELECT state, average_high FROM cities
LEFT OUTER JOIN weather 
ON name = city

SELECT state, AVG(average_high) FROM cities
LEFT OUTER JOIN weather 
ON name = city
GROUP BY state

SELECT state, AVG(average_high) FROM cities
LEFT OUTER JOIN weather 
ON name = city
GROUP BY state
ORDER BY AVG(average_high) DESC

SELECT state, average_high FROM cities
LEFT OUTER JOIN weather 
ON name = city
ORDER BY average_high DESC

SELECT state, average_high FROM cities
LEFT OUTER JOIN weather 
ON name = city
ORDER BY average_high DESC
HAVING average_high > 0 #this is wrong, HAVING is only for grouped data and connot be used in joint table?

SELECT warm_month, AVG(average_high) FROM weather
GROUP BY warm_month
HAVING AVG(average_high) > 67


INSERT INTO cities (name, state) VALUES
    ('New York City', 'NY'),
    ('Boston', 'MA'),
    ('Chicago', 'IL'),
    ('Miami', 'FL'),
    ('Dallas', 'TX'),
    ('Seattle', 'WA'),
    ('Portland', 'OR'),
    ('San Francisco', 'CA'),
    ('Los Angeles', 'CA')