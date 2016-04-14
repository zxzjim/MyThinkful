import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# Create a scatterplot of sepal length by width in the data set. Sepals
# are a type of leaf around the petals of a flower.
iris_raw = pd.read_csv('iris.data.csv')
print iris_raw.head()
features = ['sepel_len', 'sepel_width', 'pedal_len', 'pedal_width']
features_int = [i for i, x in enumerate(features)]
print features_int
iris_raw.columns = [
    'sepel_len', 'sepel_width', 'pedal_len', 'pedal_width', 'class']
print iris_raw.head()

# iris_raw[['sepel_len', 'sepel_width', 'class']].plot.scatter(
#     x='sepel_len', y='sepel_width', c=features_int)
sns.lmplot('sepel_len', 'sepel_width',
           data=iris_raw, hue='class', fit_reg=False)
plt.show()

# Pick a new point, programmatically at random.
sample = iris_raw[['sepel_len', 'sepel_width']].sample()
print sample


# Sort each point by its distance from the new point, and subset the 10
# nearest points.
iris_sepel = iris_raw[['sepel_len', 'sepel_width']]

# def cal_dist(row):
# 	return squareform(cdist([row['sepel_len'], row['sepel_width']], sample))
        # return row['sepel_len'] - row['sepel_width']


# iris_raw['distance'] = iris_sepel.apply(cal_dist, axis=1)
# print iris_raw.distance


# Determine the majority class of the subset.


# See if you can write a function called knn() that will take k as an
# argument and return the majority class for different values of k.
print iris_raw.shape
test_idx = np.random.uniform(0, 1, len(iris_raw)) <= 0.3
train_df = iris_raw[test_idx == True]
test_df = iris_raw[test_idx == False]


def knn(k, train_df, test_df, features, y_label):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_df[features], train_df[y_label])
    preds = clf.predict(test_df[features])
    return preds, clf.score(test_df[features], test_df['class'])

print knn(3, train_df, test_df, features, 'class')[1]
print knn(5, train_df, test_df, features, 'class')[1]
print knn(13, train_df, test_df, features, 'class')[1]
print knn(21, train_df, test_df, features, 'class')[1]
print knn(33, train_df, test_df, features, 'class')[1]
