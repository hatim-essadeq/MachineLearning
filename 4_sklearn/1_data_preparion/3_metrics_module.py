# ---------- Linear Regression -----------------------------------------------------------------------------------------
'''
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
#print('Mean Absolute Error Value is : ', MAEValue)
'''

from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mean_absolute_error(y_true, y_pred)

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

mean_absolute_error(y_true, y_pred)  # 0.75
mean_absolute_error(y_true, y_pred, multioutput='uniform_average')  # 0.75

mean_absolute_error(y_true, y_pred, multioutput='raw_values')  # array([0.5, 1. ])

# Import Libraries
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------

# Calculating Mean Squared Error
MSEValue = mean_squared_error(y_true, y_pred, multioutput='uniform_average')  # it can be raw_values
# print('Mean Squared Error Value is : ', MSEValue)

from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

mean_squared_error(y_true, y_pred)
mean_squared_error(y_true, y_pred, multioutput='uniform_average')

mean_squared_error(y_true, y_pred, multioutput='raw_values')

# Import Libraries
from sklearn.metrics import median_absolute_error

# ----------------------------------------------------

# Calculating Median Squared Error
MdSEValue = median_absolute_error(y_true, y_pred)
# print('Median Squared Error Value is : ', MdSEValue )

# ---------- Classification --------------------------------------------------------------------------------------------
'''
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
'''

from sklearn.metrics import confusion_matrix

y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']
y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
confusion_matrix(y_true, y_pred)

# =======================================================================

y_pred = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']
y_true = ['a', 'a', 'b', 'b', 'a', 'b', 'c', 'c', 'b', 'b']
confusion_matrix(y_true, y_pred)

# =======================================================================

y_pred = [5, 8, 9, 9, 8, 5, 5, 9, 8, 5, 9, 8]
y_true = [9, 9, 8, 8, 5, 5, 9, 5, 8, 9, 8, 5]
confusion_matrix(y_true, y_pred)

# ---------- Accuracy --------------------------------------------------------------------------------------------------
'''
#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
AccScore = accuracy_score(y_test, y_pred, normalize=False)
#print('Accuracy Score is : ', AccScore)
'''

from sklearn.metrics import accuracy_score

y_pred = [0, 2, 1, 3, 5, 3]
y_true = [0, 1, 2, 3, 5, 3]
print(accuracy_score(y_true, y_pred))  # fraction of all Trues over everything
print(accuracy_score(y_true, y_pred, normalize=False))  # number of all Trues

# ---------- F1 Score --------------------------------------------------------------------------------------------------

'''
#Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)
# f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)

F1Score = f1_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('F1 Score is : ', F1Score)
'''
from sklearn.metrics import f1_score

y_pred = [0, 2, 1, 0, 0, 1]
y_true = [0, 1, 2, 0, 1, 2]
f1_score(y_true, y_pred, average='micro')

# ---------- Recall Score : (Sensitivity) ------------------------------------------------------------------------------
'''
#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2  
# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)

RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('Recall Score is : ', RecallScore)
'''

from sklearn.metrics import recall_score

y_pred = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']
y_true = ['a', 'a', 'b', 'b', 'a', 'b', 'c', 'c', 'b', 'b']
recall_score(y_true, y_pred, average=None)

# ---------- Precision Score : (Specificity) ---------------------------------------------------------------------------
'''
#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  
# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)

PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('Precision Score is : ', PrecisionScore)
'''
from sklearn.metrics import precision_score

y_pred = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']
y_true = ['a', 'a', 'b', 'b', 'a', 'b', 'c', 'c', 'b', 'b']

precision_score(y_true, y_pred, average=None)

# ---------- Precision recall Score ------------------------------------------------------------------------------------
'''
#Calculating Precision recall Score :  
#metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=
#                                        None, warn_for = ('precision’,’recall’, ’f-score’), sample_weight=None)

PrecisionRecallScore = precision_recall_fscore_support(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('Precision Recall Score is : ', PrecisionRecallScore)
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])

precision_recall_fscore_support(y_true, y_pred, average=None)

# ---------- Precision recall Curve ------------------------------------------------------------------------------------
'''
#Calculating Precision recall Curve :  
# precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)

PrecisionValue, RecallValue, ThresholdsValue = precision_recall_curve(y_test,y_pred)
#print('Precision Value is : ', PrecisionValue)
#print('Recall Value is : ', RecallValue)
#print('Thresholds Value is : ', ThresholdsValue)
'''

import numpy as np
from sklearn.metrics import precision_recall_curve

y_pred = np.array([0, 0, 1, 1])
y_true = np.array([0.1, 0.4, 0.35, 0.8])

precision, recall, thresholds = precision_recall_curve(y_pred, y_true)

print(precision)
print(recall)
print(thresholds)

# ---------- Classification Report -------------------------------------------------------------------------------------
'''
#Calculating classification Report :  
#classification_report(y_true, y_pred, labels=None, target_names=None,sample_weight=None, digits=2, output_dict=False)

ClassificationReport = classification_report(y_test,y_pred)
#print('Classification Report is : ', ClassificationReport )
'''

from sklearn.metrics import classification_report

y_true = [0, 1, 2, 2, 2, 5]
y_pred = [0, 0, 2, 2, 1, 0]
print(classification_report(y_true, y_pred))

# ==========================================================


y_true = ['a', 'd', 'a', 'g', 'a', 'd']
y_pred = ['a', 'a', 'g', 'g', 'd', 'g']
print(classification_report(y_true, y_pred))

# ---------- Receiver Operating Characteristic -------------------------------------------------------------------------
'''
#Calculating Receiver Operating Characteristic :  
#roc_curve(y_true, y_score, pos_label=None, sample_weight=None,drop_intermediate=True)

fprValue, tprValue, thresholdsValue = roc_curve(y_test,y_pred)
#print('fpr Value  : ', fprValue)
#print('tpr Value  : ', tprValue)
#print('thresholds Value  : ', thresholdsValue)
'''

import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

# ---------- Area Under the Curve ------------------------------------------------------------------------------------
'''
#Calculating Area Under the Curve :  

fprValue2, tprValue2, thresholdsValue2 = roc_curve(y_test,y_pred)
AUCValue = auc(fprValue2, tprValue2)
#print('AUC Value  : ', AUCValue)
'''
import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

metrics.auc(fpr, tpr)
print(fpr, tpr, thresholds)

# ---------- ROC AUC Score ------------------------------------------------------------------------------------
'''
#Calculating ROC AUC Score:  
#roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)

ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
#print('ROCAUC Score : ', ROCAUCScore)
'''

import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
metrics.roc_auc_score(y, scores)

# ---------- Zero One Loss ------------------------------------------------------------------------------------
'''
#Calculating Zero One Loss:  
#zero_one_loss(y_true, y_pred, normalize = True, sample_weight = None)

ZeroOneLossValue = zero_one_loss(y_test,y_pred,normalize=False)
#print('Zero One Loss Value : ', ZeroOneLossValue )
'''
from sklearn.metrics import zero_one_loss

y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]

zero_one_loss(y_true, y_pred)

zero_one_loss(y_true, y_pred, normalize=False)
