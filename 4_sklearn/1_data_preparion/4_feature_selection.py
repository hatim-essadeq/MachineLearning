# ---------- Selection by Percentile -----------------------------------------------------------------------------------
'''
#Feature Selection by Percentile
#print('Original X Shape is ' , X.shape)
FeatureSelection = SelectPercentile(score_func = chi2, percentile=20) # score_func can = f_classif
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension
#print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support()) # show which selected columns
'''

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2

X, y = load_digits(return_X_y=True)

X.shape

X_new = SelectPercentile(score_func=chi2, percentile=10).fit_transform(X, y)

print(X_new.shape)
# ==========================================================
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile, chi2

data = load_breast_cancer()
X = data.data
y = data.target
X.shape
sel = SelectPercentile(score_func=chi2, percentile=20).fit_transform(X, y)
sel.shape

# ==========================================================
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2

X, y = load_digits(return_X_y=True)
X.shape

X_new = SelectPercentile(score_func=chi2, percentile=10)
X_new.fit(X, y)
selected = X_new.transform(X)
X_new.get_support()

# ---------- Selection by Generic --------------------------------------------------------------------------------------
'''
#Feature Selection by Generic
#print('Original X Shape is ' , X.shape)
FeatureSelection = GenericUnivariateSelect(score_func= chi2, mode= 'k_best', param=3) # score_func can = f_classif : mode can = percentile,fpr,fdr,fwe 
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
#print('X Shape is ' , X.shape)
#print('Selected Features are : ' , FeatureSelection.get_support())
'''

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import GenericUnivariateSelect, chi2

X, y = load_breast_cancer(return_X_y=True)
X.shape

transformer = GenericUnivariateSelect(score_func=chi2, mode='k_best', param=5)  # give 5 features
X_new = transformer.fit_transform(X, y)

X_new.shape

transformer.get_support()

# ---------- Selection by KBest ----------------------------------------------------------------------------------------
'''
#Feature Selection by KBest 
#print('Original X Shape is ' , X.shape)
FeatureSelection = SelectKBest(score_func= chi2 ,k=3) # score_func can = f_classif 
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
#print('X Shape is ' , X.shape)
#print('Selected Features are : ' , FeatureSelection.get_support())
'''
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

X, y = load_digits(return_X_y=True)
X.shape

X_new = SelectKBest(chi2, k=30).fit_transform(X, y)

X_new.shape

# ---------- Selection by Model ----------------------------------------------------------------------------------------
'''
#Feature Selection by FromModel
#print('Original X Shape is ' , X.shape)

from sklearn.linear_model import LinearRegression
thismodel = LinearRegression()

FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None) # make sure that thismodel is well-defined
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension
#print('X Shape is ' , X.shape)
#print('Selected Features are : ' , FeatureSelection.get_support())
'''

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

data = load_breast_cancer()
X = data.data
y = data.target

sel = SelectFromModel(RandomForestClassifier(n_estimators=20))
sel.fit(X, y)
selected_features = sel.transform(X)
sel.get_support()
