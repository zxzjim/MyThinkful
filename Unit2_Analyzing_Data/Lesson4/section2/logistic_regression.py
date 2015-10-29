# -*- coding: utf-8 -*-
import pandas as pd
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

'''
What is the probability of getting a loan from the Lending Club 
for $10,000 at an interest rate ≤ 0.12, with a FICO score of 750?

now clean the data
'''

#read the data
loansData = pd.read_csv('loansData.csv')

#clean the data
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: float(x.strip('%'))/100)
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: x.strip(' months'))
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: min(map(int, x.split('-'))))

loansData['IR_TF'] = loansData['Interest.Rate']>=.12
loansData['IR_TF'] = loansData['IR_TF'].astype(int)
loansData['Intercept'] = 1.0

#save the clean data
loansData.to_csv('loansData_clean.csv', header=True, index=False)