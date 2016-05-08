import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.decomposition import pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
print iris

X = iris.data
y = iris.target
