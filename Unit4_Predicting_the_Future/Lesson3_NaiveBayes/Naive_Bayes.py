import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.naive_bayes import GaussianNB


df_raw = pd.read_csv('ideal_weight.csv')
print df_raw.head()
print df_raw.columns


# Remove the single quotes from the column names.

new_columns = []
for col in df_raw.columns:
    col = col.replace("'", "")
    new_columns.append(col)
print new_columns
df_raw.columns = new_columns
print df_raw.head()


# Remove the single quotes from the "sex" column.

df_raw['sex'] = df_raw['sex'].apply(lambda x: x.replace("'", ""))
print df_raw.head()


# Plot the distributions of actual weight and ideal weight.
df_raw.loc[:, ['actual', 'ideal']].plot.hist(bins=30, alpha=0.8)
plt.show()

# Plot the distributions of difference in weight.
df_raw['diff'].plot.hist(bins=30, alpha=0.8)
plt.show()

sex_mapping = {'Male': 1, 'Female': 0}
df_raw['sex_mapping'] = df_raw['sex'].map(lambda x: sex_mapping[x])
print df_raw.head()
df_raw['sex_mapping'].plot.hist()
plt.show()


# Fit a Naive Bayes classifier of sex to actual weight, ideal weight, and diff.
features = df_raw.loc[:, ['actual', 'ideal', 'diff']]
Y = df_raw['sex_mapping']
clf = GaussianNB()
model = clf.fit(features, Y)

# How many points were mislabeled? How many points were there in the
# dataset, total?
pred = model.predict(features)
print "Number of mislabeled points is: %d, out of total number of %d points." % ((Y != pred).sum(), features.shape[0])
# Predict the sex for an actual weight of 145, an ideal weight of 160, and a diff of -15.
print clf.predict([[145, 160, -15]])
# Predict the sex for an actual weight of 160, an ideal weight of 145, and a diff of 15.
print clf.predict([[160, 145, 15]])
