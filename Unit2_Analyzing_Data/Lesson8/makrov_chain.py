# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:44:22 2015

@author: zhangxinzhou
"""

import pandas as pd
df = pd.DataFrame({'rainy': [.4, .7],
                   'sunny': [.6, .3]}, index = ['rainy','sunny'])
print df

df.dot(df)

df1 = pd.DataFrame({'bull': [0.9, 0.015, 0.025],
                    'bear': [0.075, 0.8, 0.25],
                    'stag': [0.025, 0.05, 0.05]}, index = ['bull', 'bear', 'stag'], columns=['bull', 'bear', 'stag'])

print df1

print df1.dot (df1)
print df1**3
print df1**5
print df1**10
print df1**100
print df1**5000
print df1**10000