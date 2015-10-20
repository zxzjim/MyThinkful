import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

#read the data
loansData = pd.read_csv('loansData.csv')

#clean the data
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: float(x.strip('%'))/100)
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: x.strip(' months'))
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: min(map(int, x.split('-'))))

#plot the histogram of FICO scores
plt.figure()
p = loansData['FICO.Score'].hist()
#plt.show('FICO_Score.png')

#create a scatterplot matrix
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
#plt.show('scatter_matrix.png')

#find the linear regression
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

y = np.matrix(intrate).transpose()
x1 = np.matrix(loanamt).transpose()
x2 = np.matrix(fico).transpose()
x = np.column_stack([x1,x2])

#create linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
print f.summary

