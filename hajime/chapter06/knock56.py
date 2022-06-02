from sklearn.metrics import *
from knock53 import *

print(classification_report(test["CATEGORY"], test_pred[1]))

"""
              precision    recall  f1-score   support

           b       0.88      0.94      0.91       556
           e       0.86      0.96      0.91       528
           m       0.93      0.58      0.71        97
           t       0.85      0.52      0.64       155

    accuracy                           0.87      1336
   macro avg       0.88      0.75      0.79      1336
weighted avg       0.87      0.87      0.86      1336

"""
