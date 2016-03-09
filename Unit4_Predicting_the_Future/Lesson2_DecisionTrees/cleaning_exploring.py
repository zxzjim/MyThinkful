import pandas as pd
import re

df_features = pd.read_csv('/Users/zhangxinzhou/DataScientist/Thinkful/Unit4_Predicting_the_Future/Lesson2_DecisionTrees/UCI HAR Dataset/features.txt', delim_whitespace=True, index_col=0, header=None)
df_raw_data = pd.read_csv('/Users/zhangxinzhou/DataScientist/Thinkful/Unit4_Predicting_the_Future/Lesson2_DecisionTrees/UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
df_activities = pd.read_csv('/Users/zhangxinzhou/DataScientist/Thinkful/Unit4_Predicting_the_Future/Lesson2_DecisionTrees/UCI HAR Dataset/activity_labels.txt', delim_whitespace=True, header=None, index_col=0)
df_features.reset_index(drop=True, inplace=True)
df_features.rename(columns={1:'labels'}, inplace=True)
print df_features

df_raw_data_t = df_raw_data.T
print df_raw_data_t

print df_features.shape
print df_raw_data_t.shape

df = pd.concat([df_features, df_raw_data_t], axis=1)
print df

df.drop_duplicates('labels', inplace=True)
print df

df = df.T
df.drop(['labels'], inplace=True)
print df

df_features = df_features.drop_duplicates().reset_index(drop=True)
print df_features

df_features = df_features.applymap(lambda e: e.replace('()', ''))
print df_features

df_features = df_features.applymap(lambda e: e.replace('-', '_'))
print df_features

df_features = df_features.applymap(lambda e: e.strip(')'))
print df_features

df_features = df_features.applymap(lambda e: e.replace(')', '_'))
print df_features

df_features = df_features.applymap(lambda e: e.replace('(','_'))
print df_features

df_features = df_features.applymap(lambda e: e.replace('BodyBody','Body'))
print df_features

df_features = df_features.applymap(lambda e: re.sub('Body|Mag','',e))
print df_features

df_features = df_features.applymap(lambda e: re.sub('mean','Mean',e))
print df_features

df_features = df_features.applymap(lambda e: re.sub('std','STD',e))
print df_features

print list(df_features['labels'])
df.columns = list(df_features['labels'])
print df

df = df.dropna()
print df