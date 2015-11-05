# -*- coding: utf-8 -*-
import pandas as pd
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

'''
What is the probability of getting a loan from the Lending Club 
for $10,000 at an interest rate ≤ 0.12, with a FICO score of 750?

now analysis the data
'''

loansData = pd.read_csv('loansData_clean.csv')
ind_vars = ['FICO.Score', 'Amount.Requested', 'Intercept']

#logistic regression model
logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars])
result = logit.fit()
coeff = result.params
print coeff

FICOScore = 750
LoanAmount = 10000



def logistic_function(FICOScore, LoanAmount, coeff):
	interest_rate = coeff['Intercept'] + coeff['FICO.Score']*FICOScore + coeff['Amount.Requested']*LoanAmount
	p = 1/(1+math.exp(coeff['Intercept'] + coeff['FICO.Score']*FICOScore + coeff['Amount.Requested']*LoanAmount))
	return p

p = logistic_function(750, 10000, coeff)
print 'The probability of getting the loan is '+str(p)

Fico = range(550, 950, 10)
p_plus = []
p_minus = []
p = []
for j in Fico:
    p_plus.append(1/(1+math.exp(coeff['Intercept'] + coeff['FICO.Score']*j + coeff['Amount.Requested']*LoanAmount)))
    p_minus.append(1/(1+math.exp(-coeff['Intercept'] - coeff['FICO.Score']*j - coeff['Amount.Requested']*LoanAmount)))
    p.append(logistic_function(j, 10000,coeff))

plt.plot(Fico, p_plus, label = 'p(x) = 1/(1+exp(b+mx))', color = 'blue')
plt.hold(True)
plt.plot(Fico, p_minus, label = 'p(x) = 1/(1+exp(-b-mx))', color = 'green')    
plt.hold(True)
plt.plot(Fico, p, 'ro', label = 'Decision for 10000 USD')
plt.legend(loc='upper right')
plt.xlabel('Fico Score')
plt.ylabel('Probability and decision, yes = 1, no = 0')
plt.show()