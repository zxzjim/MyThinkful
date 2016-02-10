
# coding: utf-8

# In[1]:

'''
1. Take the loan data and process it as you did previously to build your linear regression model.
2. Break the data-set into 10 segments following the example provided here in KFold .
3. Compute each of the performance metric (MAE, MSE or R2) for all the folds. The average would be the performance of your model.
4. Comment on each of the performance metric you obtained.
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
loansData = pd.read_csv("https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv")
loansData.head()


# In[2]:

loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: float(x.replace("%","")))
loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].map(lambda x: float(x.replace("%",""))/100)
loansData['FICO.Range'] = loansData['FICO.Range'].map(lambda x: x.split("-"))
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[0]))
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: x.replace(" months","")) 


# In[3]:

loansData.head()


# In[4]:

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# print intrate
# print loanamt
# print fico


# In[5]:

# first simple model
x = np.column_stack([loanamt,fico])
print x


# In[6]:

y = np.matrix(intrate)
# print y
# print intrate


# In[7]:

X = sm.add_constant(x)
model = sm.OLS(intrate,X)
f = model.fit()


# In[8]:

f.summary()


# In[9]:

# second model includes monthly income as another variable
income = loansData['Monthly.Income']
print income.mean()
income = income.fillna(income.mean())
x1 = pd.concat([fico,income,loanamt], axis=1)
#print x1
x1 = np.column_stack([x1[x1.columns[0]],x1[x1.columns[1]],x1[x1.columns[2]]])
print x1


# In[10]:

X = sm.add_constant(x1)
model2 = sm.OLS(intrate, X)
f = model2.fit()
f.summary()


# In[11]:

#the third model to include the debt/income ratio
ratio = loansData['Debt.To.Income.Ratio']
ratio = ratio.fillna(ratio.mean())
x2 = pd.concat([fico,income,loanamt,ratio], axis=1)
x2 = np.column_stack([x2[x2.columns[0]],x2[x2.columns[1]],x2[x2.columns[2]],x2[x2.columns[3]]])
print x2


# In[14]:

X = sm.add_constant(x2)
model3 = sm.OLS(intrate,X)
f = model3.fit()
f.summary()


# In[18]:

#=============2. Break the data-set into 10 segments following the example provided here in KFold .===========
from sklearn.cross_validation import KFold
kf = KFold(2499, 10)
for train, test in kf:
    print("Train: %s, Test: %s" % (train, test))
    X_train, X_test = X[train], X[test]
    y_train, y_test = intrate[train], intrate[test]


# In[ ]:

#==3. Compute each of the performance metric (MAE, MSE or R2) for all the folds. The average would be the performance of your model.==

