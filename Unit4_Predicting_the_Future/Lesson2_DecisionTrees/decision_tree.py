import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier

df_raw_data = pd.read_csv('samsung/samsungdata.csv')
df_train_data = pd.read_csv('samsung/samtrain.csv')
df_val_data = pd.read_csv('samsung/samval.csv')
df_test_data = pd.read_csv('samsung/samtest.csv')

print df_raw_data.head()
labels = df_train_data['activity'].unique()
print labels
label_dict = {'walk': 1, 'walkup': 2,
              'walkdown': 3, 'sitting': 4, 'standing': 5, 'laying': 6}

df_train_data['activity'] = df_train_data[
    'activity'].apply(lambda x: label_dict[x])
print df_train_data

train_data = df_train_data.iloc[:, 1:-2]
print train_data.shape
train_label = df_train_data['activity']

rfc = RandomForestClassifier(n_estimators=500, n_jobs=4, oob_score=True)
model = rfc.fit(train_data, train_label)

print rfc.oob_score_

fi = enumerate(rfc.feature_importances_)
for i in fi:
    print i

cols = train_data.columns
[(value, cols[i]) for (i, value) in fi if value > 0.04]
print train_data.head()


val_data = df_val_data.iloc[:, 1:-2]
print val_data.head()
val_label = df_val_data['activity']
val_pred = rfc.predict(val_data)
print "mean accuracy score for the validation set = %s" % (rfc.score(val_data, val_label))
