"""
学習データ，検証データ，評価データから特徴量を抽出し，
それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
"""
import string
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from knock50 import train, test, valid


def preprocessing(text):
    """記号除去，小文字化，数字を統一する前処理関数"""
    table = str.maketrans(
        string.punctuation, " " * len(string.punctuation)
    )  # string.punctuationは記号の文字列，そのそれぞれの記号をキー，スペースを値とした辞書を作成
    text = text.translate(table)  # tableにあるキーに該当したらそのキーに対応する値(スペース)に変換する．
    text = text.lower()  # 小文字化
    text = re.sub("[0-9]+", "0", text)  # 数字列を0に統一

    return text


# 前処理
train["TITLE"] = train["TITLE"].map(preprocessing)
valid["TITLE"] = valid["TITLE"].map(preprocessing)
test["TITLE"] = test["TITLE"].map(preprocessing)

# 10回以上登場したもので，unigramとbigramの両方を考える
vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

# データを入力してtf-idf行列を得る
X_train = vec_tfidf.fit_transform(train["TITLE"])
X_valid = vec_tfidf.transform(valid["TITLE"])
X_test = vec_tfidf.transform(test["TITLE"])

# データフレームに戻す
X_train = pd.DataFrame(X_train.toarray(), columns=vec_tfidf.get_feature_names_out())
X_valid = pd.DataFrame(X_valid.toarray(), columns=vec_tfidf.get_feature_names_out())
X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names_out())

# csvで出力
X_train.to_csv("X_train.txt", sep="\t", index=False)
X_valid.to_csv("X_valid.txt", sep="\t", index=False)
X_test.to_csv("X_test.txt", sep="\t", index=False)
