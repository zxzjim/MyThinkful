import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# iris_raw = pd.read_csv('iris.data.csv')
# print iris_raw.head()

# labels = ['sepel_len', 'sepel_width', 'pedal_len', 'pedal_width', 'species']

# iris_raw.columns = labels
# print iris_raw.head()
# iris_sub = iris_raw[['sepel_len', 'pedal_len', 'species']]

# sns.lmplot(x='sepel_len', y='pedal_len',
#            hue='species', data=iris_sub, fit_reg=False)
# plt.show()

# sns.lmplot('sepel_width', 'pedal_width',
#            hue='species', data=iris_raw, fit_reg=False)
# plt.show()

# sns.lmplot('sepel_len', 'sepel_width', hue='species',
#            data=iris_raw, fit_reg=False)
# plt.show()

# sns.lmplot('pedal_len', 'pedal_width', hue='species',
#            data=iris_raw, fit_reg=False)
# plt.show()


########################################## UN Dataset to do clustering ###
from scipy.cluster.vq import kmeans, vq
from pylab import plot, show


un_raw = pd.read_csv('un.csv')
print un_raw.head()
print un_raw.shape
print un_raw.dropna().shape

un_notnull = un_raw.dropna()
print un_notnull.head()

print un_notnull.dtypes

print un_notnull['country'].shape

print un_notnull.values

un_sub = un_notnull[['GDPperCapita', 'lifeFemale']].values
print un_sub

# computing K-Means with K clusters
cluster_numbers = {}
for k in range(1, 11):
    centroids, _ = kmeans(un_sub, k)
    # assign each sample to a cluster, idx=cluster index, idxdist=distance between the data points and its nearst centroid
    idx, idxdist = vq(un_sub, centroids)

    # print idx
    # print idxdist
    # print len(idx)

    results_df = pd.DataFrame({'idx': idx, 'idxdist': idxdist})
    # print results_df.head()
    cluster_numbers[k]=results_df['idxdist'].mean()
print cluster_numbers
result = pd.DataFrame(cluster_numbers.items(), columns=['cluster_numbers', 'avg_dist'])
print result
plt.plot(result['cluster_numbers'], result['avg_dist'])
plt.show()

#from above we see 3 is a good k
centroids, _ = kmeans(un_sub, 3)
# assign each sample to a cluster, idx=cluster index, idxdist=distance between the data points and its nearst centroid
idx, idxdist = vq(un_sub, centroids)

print idx
un_notnull['GDPperCapita_vs_lifeFemale_cluster'] = idx
sns.lmplot('GDPperCapita', 'lifeFemale',
           hue='GDPperCapita_vs_lifeFemale_cluster', data=un_notnull, fit_reg=False, )
for label, x, y in zip(un_notnull['region'], un_notnull['GDPperCapita'], un_notnull['lifeFemale']):
    plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.show()

plot(un_sub[idx == 0, 0], un_sub[idx == 0, 1], 'ob',
     un_sub[idx == 1, 0], un_sub[idx == 1, 1], 'or',
     un_sub[idx == 2, 0], un_sub[idx == 2, 1], 'og')
plot(centroids[:, 0], centroids[:, 1], 'sm', markersize=8)
show()
