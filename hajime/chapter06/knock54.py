from sklearn.metrics import accuracy_score
from knock53 import *

train_acc = accuracy_score(train["CATEGORY"], train_pred[1])
test_acc = accuracy_score(test["CATEGORY"], test_pred[1])

print(f"訓練データの正解率:{train_acc:.3f}")
print(f"テストデータの正解率:{test_acc:.3f}")

"""
訓練データの正解率:0.922
テストデータの正解率:0.870
"""
