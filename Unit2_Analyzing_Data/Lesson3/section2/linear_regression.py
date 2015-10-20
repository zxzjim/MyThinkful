import pandas as pd
import matplotlib.pyplot as plt

#read the data
loansData = pd.read_csv('loansData.csv')

#clean the data
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: float(x.strip('%'))/100)
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: x.strip(' months'))
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: min(map(int, x.split('-'))))

#plot the histogram of FICO scores
plt.figure()
p = loansData['FICO.Score'].hist()
plt.savefig('FICO_Score.png')

#create a scatterplot matrix
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
plt.savefig('scatter_matrix.png')