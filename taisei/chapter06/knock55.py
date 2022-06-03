from sklearn.metrics import confusion_matrix
import pandas as pd

df_y_test_ans = pd.read_table("./output/test.txt", header=None)[0]  # 正解ラベル(betmのどれか)のみ抽出
df_y_test_pred = pd.read_table("./output/knock53_test_pred.txt", header=None)[0]  # 予測結果(betmのどれか)のみ抽出

df_y_train_ans = pd.read_table("./output/train.txt", header=None)[0]
df_y_train_pred = pd.read_table("./output/knock53_train_pred.txt", header=None)[0]

print(f'学習データでの混同行列\n{confusion_matrix(df_y_train_ans, df_y_train_pred)}')
print(f'評価データでの混同行列\n{confusion_matrix(df_y_test_ans, df_y_test_pred)}')

"""結果
学習データでの混同行列
[[4427   50    2   23]
 [  24 4195    0    4]
 [  97  120  505    6]
 [ 170  118    3  928]]

評価データでの混同行列
[[534  23   0   6]
 [  9 518   1   0]
 [ 20  27  42   2]
 [ 47  31   0  74]]
"""