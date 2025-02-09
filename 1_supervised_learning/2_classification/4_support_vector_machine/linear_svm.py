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

data = loadmat('data/linear_data.mat')
# print(data1)

X = data['X']
y = data['y']

print('X:', X.shape)
print('y:', y.shape)

# data before classifying
plotData(X, y, 50)

# small C ==  UF
clfUF = svm.SVC(C=1.0, kernel='linear')
clfUF.fit(X, y.ravel())
plot_svc(clfUF, X, y)

# big C ==  OF
clfOF = svm.SVC(C=100, kernel='linear')
clfOF.fit(X, y.ravel())
plot_svc(clfOF, X, y)
