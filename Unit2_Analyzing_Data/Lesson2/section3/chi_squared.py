import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import collections

loansData = pd.read_csv('loansData.csv')
loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])
plt.figure()
plt.bar(freq.keys(), freq.values(), width = 1)
plt.show()

chi, p = stats.chisquare(freq.values())
print chi, p