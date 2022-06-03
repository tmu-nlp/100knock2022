import pandas as pd
import string
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# CountVectorizer -> BoW
# TfidfVectorizer -> tf-idf


def prep(text):
    text = "".join([i for i in text if i not in string.punctuation])
    # string.punctuation -> '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = text.lower()  # 小文字化
    text = re.sub("[0-9]+", "", text)  # 数字を削除
    return text


header_name = ['TITLE', 'CATEGORY']

train = pd.read_csv('./train.txt', header=None,
                    sep='\t', names=header_name)
valid = pd.read_csv('./valid.txt', header=None,
                    sep='\t', names=header_name)
test = pd.read_csv('./test.txt', header=None,
                   sep='\t', names=header_name)

#  concatで結合させる
df = pd.concat([train, valid, test], axis=0)
df.reset_index(drop=True, inplace=True)

# https://note.nkmk.me/python-pandas-map-applymap-apply/
# それぞれのxに対してprep(x)を返す
# 要は前処理
df["TITLE"] = df["TITLE"].map(lambda x: prep(x))

# 分割
# trainとvalidのデータのみを用いてモデルを学習するため
train_valid_d = df[:len(train)+len(valid)]
test_d = df[len(train)+len(valid):]

# dfの最小値を設定
# 使用するn_gramを1~2gramに設定
vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

# fit -> 統計量を計算
# transform -> fitの統計量に基づき正規化を実行
# fit_transform -> fitの計算とtransformの計算を連続で実行
# testのデータを用いずに統計量を獲得し，testはfitで得られた統計量に基づき正則化
# testのデータを含めてfitすると，テストデータを用いて学習しているため不適
train_valid_f = vec_tfidf.fit_transform(train_valid_d["TITLE"])
test_f = vec_tfidf.transform(test_d["TITLE"])

train_valid_vec = pd.DataFrame(
    train_valid_f.toarray(), columns=vec_tfidf.get_feature_names())
test_vec = pd.DataFrame(
    test_f.toarray(), columns=vec_tfidf.get_feature_names())

train_vec = train_valid_vec[:len(train)]
valid_vec = train_valid_vec[len(train):]

train_vec.to_csv("./train.feature.txt", sep="\t", index=False)
valid_vec.to_csv("./valid.feature.txt", sep="\t", index=False)
test_vec.to_csv("./test.feature.txt", sep="\t", index=False)

print(train_vec.head())
