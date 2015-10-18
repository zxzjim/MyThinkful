import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

loansData = pd.read_csv('loansData.csv')
loansData.dropna(inplace = True) #clean the data

loansData.boxplot(column='Amount.Requested')
plt.savefig('boxplot.png')

loansData.hist(column = 'Amount.Requested')
plt.savefig('hist.png')

plt.figure()
graph = stats.probplot(loansData['Amount.Requested'], dist = 'norm', plot = plt)
plt.savefig('qqplot.png')