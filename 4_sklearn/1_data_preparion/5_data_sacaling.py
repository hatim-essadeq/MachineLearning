# ---------- Standard Scaler -------------------------------------------------------------------------------------------
'''
# Standard Scaler for Data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)

# showing data
print('X \n', X[:10])
print('y \n', y[:10])
'''

from sklearn.preprocessing import StandardScaler

data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
scaler.fit(data)
print(scaler.mean_)
newdata = scaler.transform(data)
print(newdata)

newdata = scaler.fit_transform(data)
print(newdata)

# ---------- MinMaxScaler ----------------------------------------------------------------------------------------------
'''
#MinMaxScaler for Data

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])
'''

from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
scaler.fit(data)
print(scaler.data_range_)
print(scaler.data_min_)
print(scaler.data_max_)
newdata = scaler.transform(data)
print(newdata)

newdata = scaler.fit_transform(data)
print(newdata)

scaler = MinMaxScaler(feature_range=(1, 5))

# ---------- Normalizing Scaler ----------------------------------------------------------------------------------------
'''
#Normalizing Data

scaler = Normalizer(copy=True, norm='l2') # you can change the norm to 'l1' or 'max' 
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])
'''
from sklearn.preprocessing import Normalizer

X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]

# transformer = Normalizer(norm='l1' )

# transformer = Normalizer(norm='l2' )

transformer = Normalizer(norm='max')

transformer.fit(X)
transformer.transform(X)

# ---------- MaxAbsScaler Scaler ---------------------------------------------------------------------------------------
'''
#MaxAbsScaler Data

scaler = MaxAbsScaler(copy=True)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])
'''
from sklearn.preprocessing import MaxAbsScaler

X = [[1., 10., 2.],
     [2., 0., 0.],
     [5., 1., -1.]]
transformer = MaxAbsScaler().fit(X)
transformer
transformer.transform(X)

# ---------- Function Transforming Scaler ------------------------------------------------------------------------------
'''
#Function Transforming Data

# FunctionTransformer(func=None, inverse_func=None, validate= None,
#                     accept_sparse=False,pass_y='deprecated', check_inverse=True,
#                     kw_args=None,inv_kw_args=None)


scaler = FunctionTransformer(func = lambda x: x**2,validate = True) # or func = function1
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])
'''

from sklearn.preprocessing import FunctionTransformer

X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]


def function1(z):
    return np.sqrt(z)


FT = FunctionTransformer(func=function1)
FT.fit(X)
newdata = FT.transform(X)
print(newdata)

# ---------- Binarizer Data --------------------------------------------------------------------------------------------
'''
#Binarizing Data

scaler = Binarizer(threshold = 1.0)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])
'''

from sklearn.preprocessing import Binarizer

X = [[1., -1., -2.], [2., 0., -1.], [0., 1., -1.]]

transformer = Binarizer(threshold=1.5)
transformer.fit(X)

transformer

transformer.transform(X)

# ---------- PolynomialFeatures ----------------------------------------------------------------------------------------
'''
#Import Libraries
from sklearn.preprocessing import PolynomialFeatures
#----------------------------------------------------

#Polynomial the Data

scaler = PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])
'''

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)

poly = PolynomialFeatures(degree=2, include_bias=True)
poly.fit_transform(X)

poly = PolynomialFeatures(interaction_only=True)
poly.fit_transform(X)
