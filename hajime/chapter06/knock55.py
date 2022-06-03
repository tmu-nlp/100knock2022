from sklearn.metrics import confusion_matrix
from knock53 import *
import seaborn as sns
import matplotlib.pyplot as plt

train_con = confusion_matrix(train["CATEGORY"], train_pred[1])
test_con = confusion_matrix(test["CATEGORY"], test_pred[1])

print("訓練データの混同行列")
print(train_con)
print("テストデータの混同行列")
print(test_con)


sns.heatmap(train_con, annot=True, cmap="Greens")
plt.savefig("train_confusion_matrix.png")
plt.clf()
sns.heatmap(test_con, annot=True, cmap="Greens")
plt.savefig("test_confusion_matrix.png")

"""
訓練データの混同行列
[[4342  104    9   48]
 [  61 4165    2   11]
 [  99  123  481   11]
 [ 217  144    7  860]]
テストデータの混同行列
[[520  23   1  12]
 [ 19 506   1   2]
 [ 13  28  56   0]
 [ 40  33   2  80]]
"""
