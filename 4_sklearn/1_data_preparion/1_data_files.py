# Import Libraries
from sklearn.datasets import load_iris

# ----------------------------------------------------

#

# load iris data
IrisData = load_iris()

# X Data
X = IrisData.data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)
print('X Features are \n', IrisData.feature_names)

# y Data
y = IrisData.target
print('y Data is \n', y[:10])
print('y shape is ', y.shape)
print('y Columns are \n', IrisData.target_names)

# ----------------------------------------------------------------------------------------------------------------------

# Import Libraries
from sklearn.datasets import load_digits

# ----------------------------------------------------

# load digits data

DigitsData = load_digits()

# X Data
X = DigitsData.data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)

# y Data
y = DigitsData.target
print('y Data is \n', y[:10])
print('y shape is ', y.shape)

import matplotlib.pyplot as plt

plt.gray()

for g in range(10):
    print('Images of Number : ', g)
    plt.matshow(DigitsData.images[g])
    print('------------------------------')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# Import Libraries
from sklearn.datasets import load_boston

# ----------------------------------------------------

# load boston data

BostonData = load_boston()

# X Data
X = BostonData.data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)
print('X Features are \n', BostonData.feature_names)

# y Data
y = BostonData.target
print('y Data is \n', y[:10])
print('y shape is ', y.shape)

# ----------------------------------------------------------------------------------------------------------------------

# Import Libraries
from sklearn.datasets import load_wine

# ----------------------------------------------------

# load wine data

WineData = load_wine()

# X Data
X = WineData.data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)
print('X Features are \n', WineData.feature_names)

# y Data
y = WineData.target
print('y Data is \n', y[:10])
print('y shape is ', y.shape)
print('y Columns are \n', WineData.target_names)

# ----------------------------------------------------------------------------------------------------------------------

# Import Libraries
from sklearn.datasets import load_breast_cancer

# ----------------------------------------------------

# load breast cancer data

BreastData = load_breast_cancer()

# X Data
X = BreastData.data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)
print('X Features are \n', BreastData.feature_names)

# y Data
y = BreastData.target
print('y Data is \n', y[:10])
print('y shape is ', y.shape)
print('y Columns are \n', BreastData.target_names)

# ----------------------------------------------------------------------------------------------------------------------

# Import Libraries
from sklearn.datasets import load_diabetes

# ----------------------------------------------------

# load diabetes data

DiabetesData = load_diabetes()

# X Data
X = DiabetesData.data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)
print('X Features are \n', DiabetesData.feature_names)

# y Data
y = DiabetesData.target
print('y Data is \n', y[:10])
print('y shape is ', y.shape)

# ----------------------------------------------------------------------------------------------------------------------

# Import Libraries
from sklearn.datasets import make_regression

# ----------------------------------------------------

# load regression data

'''
X ,y = make_regression(n_samples=100, n_features=100, n_informative=10,
                       n_targets=1, bias=0.0, effective_rank=None,
                       tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                       random_state=None)
'''

X, y = make_regression(n_samples=10000, n_features=500, shuffle=True)

# X Data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)

# y Data
print('y Data is \n', y[:10])
print('y shape is ', y.shape)

# ----------------------------------------------------------------------------------------------------------------------

# Import Libraries
from sklearn.datasets import make_classification

# ----------------------------------------------------

# load classification data

'''
X, y = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2,
                           n_repeated = 0, n_classes = 2, n_clusters_per_class = 2, weights = None,
                           flip_y = 0.01, class_sep = 1.0, hypercube = True, shift = 0.0,
                           Scale() = 1.0, shuffle = True, random_state = None)
'''

X, y = make_classification(n_samples=100, n_features=20, shuffle=True)

# X Data
print('X Data is \n', X[:10])
print('X shape is ', X.shape)

# y Data
print('y Data is \n', y[:10])
print('y shape is ', y.shape)

# ----------------------------------------------------------------------------------------------------------------------

from sklearn.datasets import load_sample_image

china = load_sample_image('china.jpg')
china.dtype
china.shape

flower = load_sample_image('flower.jpg')
flower.dtype
flower.shape

import matplotlib.pyplot as plt

plt.imshow(china)
plt.imshow(flower)
