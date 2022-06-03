from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import string
import re

def score_lg(lg, x): # カテゴリのラベルを予測して、各カテゴリの予測確率を求める
  return [np.max(lg.predict_proba(x), axis=1), lg.predict(x)]

def pre(text): # 前処理
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  text = text.translate(table)  
  text = text.lower()  
  text = re.sub('[0-9]+', '0', text)  
  return text

# データの読み込み、抽出、分割と保存
df = pd.read_csv('./100knock2022/DUAN/chapter06/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
train.to_csv('./100knock2022/DUAN/chapter06/train.txt', sep='\t', index=False)
valid.to_csv('./100knock2022/DUAN/chapter06/valid.txt', sep='\t', index=False)
test.to_csv('./100knock2022/DUAN/chapter06/test.txt', sep='\t', index=False)

# データの結合と前処理
df = pd.concat([train, valid, test], axis=0)
df.reset_index(drop=True, inplace=True) 
df['TITLE'] = df['TITLE'].map(lambda x: pre(x))
# データの分割
train_valid = df[:len(train) + len(valid)]
test = df[len(train) + len(valid):]
# TF-IDFで特徴量を抽出する
vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2)) 
# ベクトル化、そしてDataFrameに変換する
X_train_valid = vec_tfidf.fit_transform(train_valid['TITLE']) 
X_test = vec_tfidf.transform(test['TITLE'])
X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vec_tfidf.get_feature_names_out())
X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names_out())

# 分割と保存
X_train = X_train_valid[:len(train)]
X_valid = X_train_valid[len(train):]
X_train.to_csv('./100knock2022/DUAN/chapter06/train.feature.txt', sep='\t', index=False)
X_valid.to_csv('./100knock2022/DUAN/chapter06/valid.feature.txt', sep='\t', index=False)
X_test.to_csv('./100knock2022/DUAN/chapter06/test.feature.txt', sep='\t', index=False)

# モデルの学習
lg = LogisticRegression(random_state=123, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])
# 予測確率を計算する
train_pred = score_lg(lg, X_train)
test_pred = score_lg(lg, X_test)

# accuracy_score()関数を使用し、正解率を学習データと評価データ上で計算する
# 第一引数にラベルを指定し、第二引数にモデルの予測結果を指定する
train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1])
test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1])
print(f'正解率（学習データ）：{train_accuracy:.6f}')
print(f'正解率（評価データ）：{test_accuracy:.6f}')

# 正解率（学習データ）：0.923257
# 正解率（評価データ）：0.886057
