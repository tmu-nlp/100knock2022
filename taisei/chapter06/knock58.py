import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()
C_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
train_accu_list = []
test_accu_list = []
valid_accu_list = []

df_X_test = pd.read_table("./output/test.feature.txt", header=0)
df_y_test_ans = pd.read_table("./output/test.txt", header=None)[0]  # 正解ラベル(betmのどれか)のみ抽出

df_X_train = pd.read_table("./output/train.feature.txt", header=0)
df_y_train_ans = pd.read_table("./output/train.txt", header=None)[0]

df_X_valid = pd.read_table("./output/valid.feature.txt", header=0)
df_y_valid_ans = pd.read_table("./output/valid.txt", header=None)[0]

for c in C_list:
    lr = LogisticRegression(random_state=0, max_iter = 10000, C=c)
    lr.fit(df_X_train, df_y_train_ans)
    train_accu_list.append(accuracy_score(df_y_train_ans, lr.predict(df_X_train)))
    test_accu_list.append(accuracy_score(df_y_test_ans, lr.predict(df_X_test)))
    valid_accu_list.append(accuracy_score(df_y_valid_ans, lr.predict(df_X_valid)))

end = time.time()
print(end - start)
print(C_list)
print(train_accu_list)
print(test_accu_list)
print(valid_accu_list)
plt.plot(C_list, train_accu_list, label="train")
plt.plot(C_list, test_accu_list, label="test")
plt.plot(C_list, valid_accu_list, label="valid")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("./output/knock58_output.png")
#実行時間 594.9904170036316[s]
