import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pylab as pl

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

# out of bag score:
print rfc.oob_score_

# find the important features
fi = enumerate(rfc.feature_importances_)
for i in fi:
    print i

#????? choose the features that are > 0.04 significant
cols = train_data.columns
[(value, cols[i]) for (i, value) in fi if value > 0.04]
print cols


df_val_data['activity'] = df_val_data[
    'activity'].apply(lambda x: label_dict[x])
val_data = df_val_data.iloc[:, 1:-2]
print val_data.head()
val_label = df_val_data['activity']
print val_label
val_pred = rfc.predict(val_data)
print "mean accuracy score for the validation set = %s" % (rfc.score(val_data, val_label))


df_test_data['activity'] = df_test_data[
    'activity'].apply(lambda x: label_dict[x])
test_data = df_test_data.iloc[:, 1:-2]
test_label = df_test_data['activity']
test_pred = rfc.predict(test_data)
print "mean accuracy score for the test set= %s" % (rfc.score(test_data, test_label))

test_cm = skm.confusion_matrix(test_label, test_pred)
pl.matshow(test_cm)
pl.colorbar()
pl.show()
