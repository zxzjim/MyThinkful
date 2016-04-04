import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist, cdist
plt.style.use('ggplot')

#Create a scatterplot of sepal length by width in the data set. Sepals are a type of leaf around the petals of a flower.
iris_raw = pd.read_csv('iris.data.csv')
print iris_raw.head()
iris_raw.columns = ['sepel_len', 'sepel_width', 'pedal_len', 'pedal_width', 'class']
print iris_raw.head()

iris_raw[['sepel_len', 'sepel_width']].plot.scatter(x='sepel_len', y='sepel_width')
plt.show()

#Pick a new point, programmatically at random.
sample = iris_raw[['sepel_len', 'sepel_width']].sample()
print type(sample)


#Sort each point by its distance from the new point, and subset the 10 nearest points.
iris_sepel = iris_raw[['sepel_len', 'sepel_width']]

def cal_dist(row):
	return squareform(cdist([row['sepel_len'], row['sepel_width']], sample))
	#return row['sepel_len'] - row['sepel_width']


iris_raw['distance'] = iris_sepel.apply(cal_dist, axis=1)
print iris_raw.distance


#Determine the majority class of the subset.




#See if you can write a function called knn() that will take k as an argument and return the majority class for different values of k.



