import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def preprocessing(text):
    """前処理"""
    text = text.lower()  # 小文字化
    text = re.sub(r'[0-9]', "", text)  # 数字の消去
    text = re.sub(r'[^a-z\s]', "", text)  # 空白以外の記号の消去
    return text


# ファイル読み込み
train_ch = pd.read_csv("train.txt", sep='\t')
valid_ch = pd.read_csv("valid.txt", sep='\t')
test_ch = pd.read_csv("test.txt", sep='\t')

# 前処理しやすいように合成
df = pd.concat([train_ch, valid_ch, test_ch], axis=0)
df.reset_index(drop=True, inplace=True)

# 全ての値に関数を適用
df["TITLE"] = df["TITLE"].map(lambda x: preprocessing(x))

# print(df.head())

# データ分割
train_valid = df[:len(train_ch) + len(valid_ch)]
test = df[len(train_ch) + len(valid_ch):]

# 特徴抽出 (Tf-idf, stop-words)
cv = TfidfVectorizer(min_df=10, ngram_range=(1, 2), stop_words="english")
train_valid_cv = cv.fit_transform(train_valid["TITLE"])
test_cv = cv.transform(test["TITLE"])

# 特徴抽出 (Count, stop-words)
# cv = CountVectorizer(stop_words="english")
# train_valid_cv = cv.fit_transform(train_valid['TITLE'])  # testの情報は使わない
# test_cv = cv.transform(test['TITLE'])

# ↑の時点でそれぞれベクトル化されて配列の形をとっている
# ここでベクトルをデータフレームに変換する
# numpyにあった配列に toarray() を使って変換
train_valid_cv = pd.DataFrame(
    train_valid_cv.toarray(), columns=cv.get_feature_names())
test_cv = pd.DataFrame(test_cv.toarray(), columns=cv.get_feature_names())

# データ分割
train = train_valid_cv[:len(train_ch)]
valid = train_valid_cv[len(train_ch):]

# 出力
train.to_csv("train.feture.txt", sep='\t', index=False)
valid.to_csv("valid.feture.txt", sep='\t', index=False)
test_cv.to_csv("test.feture.txt", sep='\t', index=False)

print(train.head())
