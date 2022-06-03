from sklearn.metrics import accuracy_score
import pandas as pd

df_y_test_ans = pd.read_table("./output/test.txt", header=None)[0]  # 正解ラベル(betmのどれか)のみ抽出
df_y_test_pred = pd.read_table("./output/knock53_test_pred.txt", header=None)[0]  # 予測結果(betmのどれか)のみ抽出

df_y_train_ans = pd.read_table("./output/train.txt", header=None)[0]
df_y_train_pred = pd.read_table("./output/knock53_train_pred.txt", header=None)[0]

print(f'学習データでの正解率：{accuracy_score(df_y_train_ans, df_y_train_pred)}')
print(f'評価データでの正解率：{accuracy_score(df_y_test_ans, df_y_test_pred)}')

"""結果
学習データでの正解率：0.9421851574212894
評価データでの正解率：0.8755622188905547
"""