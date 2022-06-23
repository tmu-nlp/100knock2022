import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import KeyedVectors
import pickle
import torch
import string


def transform_w2v(text):

    table = str.maketrans(string.punctuation, ' ' *
                          len(string.punctuation))  # 記号削除のトランスフォーマー
    words = text.translate(table).split()  # 記号を削除
    vec = []
    for word in words:
        if word in model:
            vec.append(model[word])

    return torch.tensor(sum(vec) / len(vec))


# 記事の読み込み
df = pd.read_csv('./data/newsCorpora.csv', sep='\t', header=None, names=[
                 'ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# 情報源の抽出
flag = df['PUBLISHER'].isin(
    ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])
df = df.loc[flag]

# データの分割(ランダム)
train_data, other_data = train_test_split(
    df, test_size=0.2, shuffle=True, random_state=42)  # 訓練データ8, その他2
valid_data, test_data = train_test_split(
    other_data, test_size=0.5, shuffle=True, random_state=42)  # 検証データ1, テストデータ1

# ファイルへ書き出し
train_data.to_csv('./data/train.txt', sep='\t', index=False)
valid_data.to_csv('./data/valid.txt', sep='\t', index=False)
test_data.to_csv('./data/test.txt', sep='\t', index=False)


# word2vecのモデル読み込み
model = KeyedVectors.load_word2vec_format(
    './data/GoogleNews-vectors-negative300.bin.gz', binary=True)

# pickle保存
with open('GoogleNews-vectors.pkl', 'wb') as f:
    pickle.dump(model, f)


# 特徴ベクトル(data)
X_train = torch.stack([transform_w2v(text) for text in train_data['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid_data['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test_data['TITLE']])

torch.save(X_train, 'X_train.pt')
torch.save(X_valid, 'X_valid.pt')
torch.save(X_test, 'X_test.pt')


# 正解ラベルベクトル(target)
d = {'b': 0, 't': 1, 'e': 2, 'm': 3}  # ビジネス、科学技術、エンターテインメント、健康
y_train = torch.tensor(np.array(train_data.loc[:, 'CATEGORY'].replace(d)))
y_valid = torch.tensor(np.array(valid_data.loc[:, 'CATEGORY'].replace(d)))
y_test = torch.tensor(np.array(test_data.loc[:, 'CATEGORY'].replace(d)))

torch.save(y_train, 'y_train.pt')
torch.save(y_valid, 'y_valid.pt')
torch.save(y_test, 'y_test.pt')
