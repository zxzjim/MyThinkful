import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9]
plt.boxplot(x)
plt.savefig("boxplot.png")

plt.figure()
plt.hist(x, histtype = 'bar')
plt.savefig("hist.png")

plt.figure()
graph = stats.probplot(x, dist = "norm", plot = plt)
plt.savefig("qqplot.png")