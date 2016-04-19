import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


iris_raw = pd.read_csv('iris.data.csv')
print iris_raw.head()

labels = ['sepel_len', 'sepel_width', 'pedal_len', 'pedal_width', 'species']
iris_raw.columns = labels
print iris_raw.head()

species = iris_raw.species.unique()
print species
species_dict = dict(zip(species, range(0, len(species))))
print species_dict

iris_raw['class'] = iris_raw['species'].apply(lambda x: species_dict[x])
print iris_raw

iris_sub = iris_raw[['sepel_len', 'pedal_len', 'species']]

sns.lmplot(x='sepel_len', y='pedal_len',
           hue='species', data=iris_sub, fit_reg=False)
plt.show()

sns.lmplot('sepel_width', 'pedal_width',
           hue='species', data=iris_raw, fit_reg=False)
plt.show()

sns.lmplot('sepel_len', 'sepel_width', hue='species',
           data=iris_raw, fit_reg=False)
plt.show()

sns.lmplot('pedal_len', 'pedal_width', hue='species',
           data=iris_raw, fit_reg=False)
plt.show()


from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

for k in range(1, 31):
    svc = svm.SVC(kernel='linear', C=k)
    X = iris_raw[['sepel_width', 'sepel_len']].values
    y = iris_raw['class'].values
    svc.fit(X, y)
    plot_estimator(svc, X, y)
    plt.show()
