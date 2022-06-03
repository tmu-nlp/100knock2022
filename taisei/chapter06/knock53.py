from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

with open("./output/knock52_lr_model", "rb") as f:
    lr = pickle.load(f)

# まずテストデータに処理してる
X_test = pd.read_table("./output/test.feature.txt", header=0)
y_test_ans = pd.read_table("./output/test.txt", header=None)[0]

y_test_pred = lr.predict(X_test) #予測カテゴリ
y_test_prob = lr.predict_proba(X_test)  #データごとの各カテゴリに分類される確率

df_y_test_pred = pd.DataFrame(data=y_test_pred)
df_y_test_prob = pd.DataFrame(data=y_test_prob)

#予測カテゴリと、分類される確率をconcatで横に結合
df_y_test = pd.concat([df_y_test_pred, df_y_test_prob], axis=1) #形式 → カテゴリ b e m t



# 訓練データに対しても同様の処理
X_train = pd.read_table("./output/train.feature.txt", header=0)
y_train_ans = pd.read_table("./output/train.txt", header=None)[0]

y_train_pred = lr.predict(X_train)
y_train_prob = lr.predict_proba(X_train)

df_y_train_pred = pd.DataFrame(data=y_train_pred)
df_y_train_prob = pd.DataFrame(data=y_train_prob)

df_y_train = pd.concat([df_y_train_pred, df_y_train_prob], axis=1) #形式 → カテゴリ b e m t


# 検証データに対しても同様の処理(58と59で使う)
X_valid = pd.read_table("./output/valid.feature.txt", header=0)
y_valid_ans = pd.read_table("./output/valid.txt", header=None)[0]

y_valid_pred = lr.predict(X_valid)
y_valid_prob = lr.predict_proba(X_valid)

df_y_valid_pred = pd.DataFrame(data=y_valid_pred)
df_y_valid_prob = pd.DataFrame(data=y_valid_prob)

df_y_valid = pd.concat([df_y_valid_pred, df_y_valid_prob], axis=1) #形式 → カテゴリ b e m t

if __name__ == "__main__":
    df_y_test.to_csv("./output/knock53_test_pred.txt", sep="\t", index=False, header=None) 
    df_y_train.to_csv("./output/knock53_train_pred.txt", sep="\t", index=False, header=None)
    df_y_valid.to_csv("./output/knock53_valid_pred.txt", sep="\t", index=False, header=None)
