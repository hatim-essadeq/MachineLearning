# Import Libraries

# ----------------------------------------------------


# Cleaning data

'''
impute.SimpleImputer(missing_values=nan, strategy='mean’, fill_value=None, verbose=0, copy=True)


ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)

#X Data
print('X Data is \n' , X[:10])

#y Data
print('y Data is \n' , y[:10])
'''

# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
# Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer

# ----------------------------------------------------

# load breast cancer data

BreastData = load_breast_cancer()

# X Data
X = BreastData.data

# y Data
y = BreastData.target

# ----------------------------------------------------
# Cleaning data

'''
impute.SimpleImputer(missing_values=nan, strategy='mean’, fill_value=None, verbose=0, copy=True)
'''

ImputedModule = SimpleImputer(missing_values=np.nan, strategy='mean')
ImputedX = ImputedModule.fit(X)

X = ImputedX.transform(X)

# X Data
print('X Data is \n', X[:10])

# y Data
print('y Data is \n', y[:10])

# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.impute import SimpleImputer

data = [[1, 2, np.nan],
        [3, np.nan, 1],
        [5, np.nan, 0],
        [np.nan, 4, 6],
        [5, 0, np.nan],
        [4, 5, 5]]

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(data)

modifieddata = imp.transform(data)
print(modifieddata)

# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.impute import SimpleImputer

data = [[1, 2, np.nan],
        [3, np.nan, 1],
        [5, np.nan, 0],
        [np.nan, 4, 6],
        [5, 0, np.nan],
        [4, 5, 5]]

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp = imp.fit(data)

modifieddata = imp.transform(data)
print(modifieddata)

# ----------------------------------------------------------------------------------------------------------------------

from sklearn.impute import SimpleImputer

data = [[1, 2, 0],
        [3, 0, 1],
        [5, 0, 0],
        [0, 4, 6],
        [5, 0, 0],
        [4, 5, 5]]

imp = SimpleImputer(missing_values=0, strategy='mean')
imp = imp.fit(data)

modifieddata = imp.transform(data)
print(modifieddata)
