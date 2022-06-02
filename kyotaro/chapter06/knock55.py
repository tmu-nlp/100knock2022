from sklearn.metrics import confusion_matrix
from knock54 import train_ans, test_ans, train_myans, test_myans

# 正解率同様にして作る
train_confusion = confusion_matrix(train_ans, train_myans)
test_confusion = confusion_matrix(test_ans, test_myans)

print("train's confusion matrix = ")
print(train_confusion)
print("test's confusion matrix = ")
print(test_confusion)
