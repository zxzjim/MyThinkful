# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:45:56 2015

@author: zhangxinzhou
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('LoanStats3b.csv', header=1, low_memory=False)

#converts string to datetime object
df['issue_d_format'] = pd.to_datetime(df['issue_d'])
dfts = df.set_index('issue_d_format')

'''why does\'t this work? 
date_summary = dfts.groupby(df.issue_d_format).count()
'''
year_month_summary = dfts.groupby(lambda x: x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

loan_count_summary.plot()

sm.graphics.tsa.plot_acf(loan_count_summary)
sm.graphics.tsa.plot_pacf(loan_count_summary)

