CREATE TABLE weather (city text, year integer, warm_month text, cold_month text, average_high integer);
insert into weather (city, year, warm_month, cold_month, average_high) values
	('Boston', 2013, 'July', 'January', 59),
	('Chicago', 2013, 'July', 'January', 59),
	('Miami', 2013, 'August', 'January', 84),
	('Dallas', 2013, 'July', 'January', 77),
	('Seattle', 2013, 'July', 'January', 61),
	('Portland', 2013, 'July', 'December', 63),
	('San Francisco', 2013, 'September', 'December', 64),
	('Los Angeles', 2013, 'September', 'December', 75);

select city, average_high from weather where cold_month == 'December'

select city from weather where warm_month == 'July' and not cold_month == 'January'

update weather set city = 'San Jose' where city = 'San Francisco';

update weather set city = 'San Francisco', average_high = 65 where city = 'Los Angeles';

select city, average_high from weather where city like 'San%';

delect from weather;