'''
本章ではNNで四カテゴリ分類モデルを実装
70. 単語ベクトルの和による特徴量
i番目の事例の記事見出しを、その見出しに含まれる単語のベクトルの平均で表現したものがx_iである。
単語ベクトルは、knock60の300次元の単語ベクトルを使用
各データ別に特徴量行列とそれに対応するラベルベクトルをファイルに保存
'''

import pandas as pd
from gensim.models import KeyedVectors
import torch
import datetime

start = datetime.datetime.now()
print(f'running start: {start}')
# knock50で作ったデータセットを読み込む
train = pd.read_table('../chapter06/train_re.txt', names=['CATEGORY', 'TITLE'])
# print(train.shape)   #(10672, 2)
valid = pd.read_table('../chapter06/valid_re.txt', names=['CATEGORY', 'TITLE'])
test = pd.read_table('../chapter06/test_re.txt', names=['CATEGORY', 'TITLE'])

model = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
def get_w2v(sent):
    words = sent.strip().split()
    vecs = [model[word] for word in words if word in model]
    return torch.tensor(sum(vecs)/len(vecs))
    # sum()は列ごとに足し算、len()は行数に当たる。
    # torch.Size([300]) センテンスごとに平均ベクトルを獲得



# make feature vector for each title
X_train = torch.stack([get_w2v(sent) for sent in train['TITLE']])
X_valid = torch.stack([get_w2v(sent) for sent in valid['TITLE']])
X_test = torch.stack([get_w2v(sent) for sent in test['TITLE']])
# print(X_train.size())

# make feature vectors for label
category_dic = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dic[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dic[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dic[x]).values)
# print(y_train.size())

end = datetime.datetime.now()
print(f'running end: {end}')
print(f'time used: {end-start}s')

if __name__ == '__main__':
    # save data
    torch.save(X_train, 'X_train.pt')
    torch.save(X_valid, 'X_valid.pt')
    torch.save(X_test, 'X_test.pt')
    torch.save(y_train, 'y_train.pt')
    torch.save(y_valid, 'y_valid.pt')
    torch.save(y_test, 'y_test.pt')


'''
running start: 2022-06-23 19:41:16.398009
torch.Size([10672, 300]) -> len(train['title']),300)
torch.Size([10672])
running end: 2022-06-23 19:42:52.797267
time used: 0:01:36.399258s
'''