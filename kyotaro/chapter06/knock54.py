from sklearn.metrics import accuracy_score
import pandas as pd
from knock53 import train_predict, test_predict


# 正解のタグと自分で予測したタグが必要

# 正解のタグ
train_ans = pd.read_csv("train.txt", sep="\t")["CATEGORY"]
test_ans = pd.read_csv("test.txt", sep="\t")["CATEGORY"]

# 自分で予測したタグ
train_myans = train_predict[1]
test_myans = test_predict[1]

# 正解率の判定
train_accuracy = accuracy_score(train_ans, train_myans)
test_accuracy = accuracy_score(test_ans, test_myans)

# print("train accuracy = ", train_accuracy)
# print("test accuracy = ", test_accuracy)
