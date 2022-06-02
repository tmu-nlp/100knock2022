"""
正解率は多クラス分類であってもただ1つだけ計算される。
再現率・適合率・F1スコアはクラスそれぞれについて計算される。
そのためクラスごとの結果を1つにまとめる必要がある。
〇〇率のマクロ平均：クラスごとに〇〇率を計算し、それらを単純に平均したもの
〇〇率のマイクロ平均：全クラスで一斉に〇〇率を計算する。正解率と一致する？
"""

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

df_y_test_ans = pd.read_table("./output/test.txt", header=None)[0]  # 正解ラベル(betmのどれか)のみ抽出
df_y_test_pred = pd.read_table("./output/knock53_test_pred.txt", header=None)[0]  # 予測結果(betmのどれか)のみ抽出

print(f'適合率[b e m t]：{precision_score(df_y_test_ans, df_y_test_pred, average=None)}\
    マイクロ平均{precision_score(df_y_test_ans, df_y_test_pred, average="micro")}\
    マクロ平均{precision_score(df_y_test_ans, df_y_test_pred, average="macro")}')

print(f'再現率[b e m t]：{recall_score(df_y_test_ans, df_y_test_pred, average=None)}\
    マイクロ平均{recall_score(df_y_test_ans, df_y_test_pred, average="micro")}\
    マクロ平均{recall_score(df_y_test_ans, df_y_test_pred, average="macro")}' )

print(f'F1スコア[b e m t]：{f1_score(df_y_test_ans, df_y_test_pred, average=None)}\
    マイクロ平均{f1_score(df_y_test_ans, df_y_test_pred, average="micro")}\
    マクロ平均{f1_score(df_y_test_ans, df_y_test_pred, average="macro")}')

"""結果
適合率[b e m t]：[0.87540984 0.86477462 0.97674419 0.90243902]    マイクロ平均0.8755622188905547    マクロ平均0.9048419177190714
再現率[b e m t]：[0.94849023 0.98106061 0.46153846 0.48684211]    マイクロ平均0.8755622188905547    マクロ平均0.7194828509420218
F1スコア[b e m t]：[0.91048593 0.91925466 0.62686567 0.63247863]    マイクロ平均0.8755622188905547    マクロ平均0.7722712240023383
"""
#これをsklearn.metricsのcalssification_reportを使って
# print(classification_report(df_y_test_ans, df_y_test_pred))で一発でできる
