
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler


# In[2]:

iris = datasets.load_iris()


# In[3]:

print iris


# In[4]:

data_df = pd.DataFrame(iris.data)
class_df = pd.DataFrame(iris.target)


# In[5]:

iris_df = pd.concat([data_df, class_df], axis=1)

iris_df.columns = ['sepel_len', 'sepel_width', 'pedal_len', 'pedal_width', 'class']


# In[6]:

iris_df.head()


# In[7]:

iris_df.tail()


# In[8]:

test_idx = np.random.uniform(0,1,len(iris_df))<=.3
print test_idx
train_df = iris_df[test_idx == False]
test_df = iris_df[test_idx == True]


# In[9]:

train_df.head()


# In[10]:

test_df.head()


# In[11]:

sns.lmplot('pedal_len', 'pedal_width', data=train_df, hue='class', fit_reg=False)


# In[12]:

from sklearn.neighbors import KNeighborsClassifier


# In[13]:

clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(train_df[['sepel_len', 'sepel_width', 'pedal_len', 'pedal_width']], train_df['class'])
clf.score(test_df[['sepel_len', 'sepel_width', 'pedal_len', 'pedal_width']], test_df['class'])


# In[14]:

train_df.iloc[:, 0:4].head()


# In[15]:

sklearn_LDA = LDA(n_components=2)
train_r = sklearn_LDA.fit_transform(train_df.iloc[:, 0:4],train_df['class'])


# In[16]:

train_df_r=pd.DataFrame(train_r,columns=['feature1', 'feature2'])


# In[17]:

train_df_r = pd.concat([train_df_r,train_df['class'].reset_index(drop=True)], axis=1 )


# In[18]:

train_df_r.head()


# In[19]:

test_r = sklearn_LDA.fit_transform(test_df.iloc[:, 0:4], test_df['class'])
test_df_r = pd.DataFrame(test_r, columns=['feature1', 'feature2'])
test_df_r = pd.concat([test_df_r,test_df['class'].reset_index(drop=True)], axis=1)
test_df_r.head()


# In[20]:

clf_r = KNeighborsClassifier(n_neighbors=9)
clf_r.fit(train_df_r.iloc[:, 0:2], train_df_r['class'])
clf_r.score(test_df_r.iloc[:, 0:2], test_df_r['class'])


# In[21]:

train_df_r.columns=['feature1', 'feature2', 'class']
sns.lmplot(x='feature1', y='feature2', data=train_df_r, hue='class', fit_reg=False)


# In[ ]:



