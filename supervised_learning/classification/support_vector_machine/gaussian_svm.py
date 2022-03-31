import pandas as pd

import seaborn as sns

from supervised_learning.classification.support_vector_machine.draw_plt import plotData, plot_svc

sns.set_context('notebook')
sns.set_style('white')

from scipy.io import loadmat
from sklearn import svm

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

# Example 1

data1 = loadmat('./data/gaussian_data1.mat')
print(data1.keys())

X1 = data1['X']
y1 = data1['y']

print('X1:', X1.shape)
print('y1:', y1.shape)

plotData(X1, y1, 8)

# apply SVM
clf1 = svm.SVC(C=50, kernel='rbf', gamma=6)
clf1.fit(X1, y1.ravel())
plot_svc(clf1, X1, y1)

# ---------------------------------------------------------

# Example 2


data2 = loadmat('./data/gaussian_data2.mat')
# print(data2.keys())

X2 = data2['X']
y2 = data2['y']

print('X2:', X2.shape)
print('y2:', y2.shape)

plotData(X2, y2, 30)

clf2 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma=10)
clf2.fit(X2, y2.ravel())
plot_svc(clf2, X2, y2)
