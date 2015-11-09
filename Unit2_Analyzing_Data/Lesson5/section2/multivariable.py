import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('LoanStats3c.csv', header=1)

df_inc = df.annual_inc.dropna(0)
df_ir = df.int_rate.dropna(0)
df_ir = df_ir.map(lambda x: float(x.strip('%'))/100)


x = df_inc
x = sm.add_constant(x)
y = df_ir

model = sm.OLS(y, x)
est = model.fit()
print est.summary()

#output the first modeling
plt.scatter(df_inc, df_ir, alpha=.3)
plt.xlabel('annual income')
plt.ylabel('interest rate')
inc_linspace = np.linspace(df_inc.min(), df_inc.max(), 100)

plt.plot(inc_linspace, est.params[0] + est.params[1] * inc_linspace, 'b')
plt.savefig('inc_ir.jpg')

#adding home_ownership to the model
df_ho = df.home_ownership.dropna(0)
df_ho = df_ho == 'RENT'
df_ho = df_ho.astype(int)

df_est = pd.DataFrame([df_ir, df_inc, df_ho])
df_est = df_est.transpose()

model = smf.ols(formula='int_rate ~ annual_inc * home_ownership', data=df_est)
est = model.fit()
print est.summary()

#output the second modeling
plt.scatter(df_inc, df_ir, alpha=.3)
plt.xlabel('annual income * home ownership')
plt.ylabel('interest rate')
inc_linspace = np.linspace(df_inc.min(), df_inc.max(), 100)

plt.plot(inc_linspace, est.params[0] + est.params[1] * inc_linspace + est.params[2] * 0 + est.params[3] * 0 * inc_linspace, 'r')
plt.plot(inc_linspace, est.params[0] + est.params[1] * inc_linspace + est.params[2] * 1 + est.params[3] * 1 * inc_linspace, 'g')

plt.savefig('inc*ho_ir.jpg')
