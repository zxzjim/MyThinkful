import pandas as pd
import scipy.stats as stats

df = pd.read_table('Alcohol_Tobacco.txt')
aRange = df['Alcohol'].max() - df['Alcohol'].min()
aMean = df['Alcohol'].mean()
aMedian = df['Alcohol'].median()
aMode = stats.mode(df['Alcohol'])
aVar = df['Alcohol'].var()
aStd = df['Alcohol'].std()

tRange = df['Tobacco'].max() - df['Tobacco'].min()
tMean = df['Tobacco'].mean()
tMedian = df['Tobacco'].median()
tMode = stats.mode(df['Tobacco'])
tVar = df['Tobacco'].var()
tStd = df['Tobacco'].std()
print "The range for the Alcohol is {0}, mean is {1}, median is {2}, mode is {3}, variance is {4} and standard deviation is {5}\n".format(aRange, aMean, aMedian, aMode, aVar, aStd)
print "The range for the Tobacco is {0}, mean is {1}, median is {2}, mode is {3}, variance is {4} and standard deviation is {5}\n".format(tRange, tMean, tMedian, tMode, tVar, tStd)
