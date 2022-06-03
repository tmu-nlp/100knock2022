from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import GridSearchCV
import pandas as pd
import time

start = time.time()

df_X_train = pd.read_table("./output/train.feature.txt", header=0)
df_y_train_ans = pd.read_table("./output/train.txt", header=None)[0]

df_X_test = pd.read_table("./output/test.feature.txt", header=0)
df_y_test_ans = pd.read_table("./output/test.txt", header=None)[0]

df_X_valid = pd.read_table("./output/valid.feature.txt", header=0)
df_y_valid_ans = pd.read_table("./output/valid.txt", header=None)[0]

best_model = None #検証データでの正解率が一番高いモデルを保持
best_accu_valid = 0 #上の時の正解率を保持
accu_valid_list = [] #探索したパラメータとその時の正解率のタプルを保持
best_model_name = "" #ベストモデルのアルゴリズム名
best_param = "" #ベストモデルのパラメータ

#ロジスティック回帰
#ハイパーパラメータは正則化パラメータとする
C = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
for c in C:
    lr = LogisticRegression(random_state=0, max_iter = 10000, C=c)
    lr.fit(df_X_train, df_y_train_ans)
    accu_valid = accuracy_score(df_y_valid_ans, lr.predict(df_X_valid)) # 検証データでの正解率
    accu_valid_list.append((f'C={c}', accu_valid))
    if best_accu_valid < accu_valid: #検証データにおいて正解率が最も高いモデルを探す
        best_model_name = "ロジスティック回帰"
        best_model = lr
        best_accu_valid = accu_valid
        best_param = c


#ランダムフォレスト
#ハイパーパラメータは決定木の数、決定木の最大深さとする
N_ESTIMA = [5, 10, 50, 100] #デフォは10
MAX_DEPTH = [5, 10, 50, None] #デフォは無限
for n_estima in N_ESTIMA:
    for max_dep in MAX_DEPTH:
        rfc = RandomForestClassifier(random_state=0, n_estimators=n_estima, max_depth=max_dep)
        rfc.fit(df_X_train, df_y_train_ans)
        accu_valid = accuracy_score(df_y_valid_ans, rfc.predict(df_X_valid))
        accu_valid_list.append((f'N_ESTIMA={n_estima} MAX_DEPTH={max_dep}', accu_valid))
        if best_accu_valid < accu_valid:
            best_model_name = "ランダムフォレスト"
            best_model = rfc
            best_accu_valid = accu_valid
            best_param = (n_estima, max_dep)
print(accu_valid_list)

print(f'検証データでの正解率が最大の学習アルゴリズム：{best_model_name}')
print(f'その時のパラメータ：{best_param}')
print(f'上記のパラメータの時の評価データでの正解率：{accuracy_score(df_y_test_ans, best_model.predict(df_X_test))}')
end = time.time()
print(f'実行時間{end - start}')

"""結果
[('C=0.01', 0.7361319340329835), ('C=0.1', 0.7646176911544228), ('C=1.0', 0.868815592203898), ('C=10.0', 0.9002998500749625), ('C=100.0', 0.9062968515742129), ('C=1000.0', 0.9047976011994003),
 ('N_ESTIMA=5 MAX_DEPTH=5', 0.4767616191904048), ('N_ESTIMA=5 MAX_DEPTH=10', 0.5644677661169415), ('N_ESTIMA=5 MAX_DEPTH=50', 0.717391304347826), ('N_ESTIMA=5 MAX_DEPTH=None', 0.7721139430284858), 
 ('N_ESTIMA=10 MAX_DEPTH=5', 0.5389805097451275), ('N_ESTIMA=10 MAX_DEPTH=10', 0.6529235382308846), ('N_ESTIMA=10 MAX_DEPTH=50', 0.7286356821589205), ('N_ESTIMA=10 MAX_DEPTH=None', 0.7661169415292354), 
 ('N_ESTIMA=50 MAX_DEPTH=5', 0.6499250374812594), ('N_ESTIMA=50 MAX_DEPTH=10', 0.6896551724137931), ('N_ESTIMA=50 MAX_DEPTH=50', 0.7436281859070465), ('N_ESTIMA=50 MAX_DEPTH=None', 0.7886056971514243), 
 ('N_ESTIMA=100 MAX_DEPTH=5', 0.6559220389805097), ('N_ESTIMA=100 MAX_DEPTH=10', 0.7181409295352323), ('N_ESTIMA=100 MAX_DEPTH=50', 0.7428785607196402), ('N_ESTIMA=100 MAX_DEPTH=None', 0.7946026986506747)]
検証データでの正解率が最大の学習アルゴリズム：ロジスティック回帰
その時のパラメータ：100.0
上記のパラメータの時の評価データでの正解率：0.9137931034482759
実行時間527.1090862751007
"""
