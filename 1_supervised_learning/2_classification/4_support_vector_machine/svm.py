import numpy as np
from scipy.io import loadmat
from sklearn import svm

spam_train = loadmat('./data/train_data.mat')
spam_test = loadmat('./data/test_data.mat')

# print(spam_train)
# print(spam_test)

#  Training
X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

print(X.shape, y.shape, Xtest.shape, ytest.shape)

svc = svm.SVC()
svc.fit(X, y)

# Testing
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))
