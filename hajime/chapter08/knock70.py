import pandas as pd
import pickle
import gensim
import string
import torch


def trans_word2vec(text):
    table = str.maketrans(string.punctuation, ' ' *
                          len(string.punctuation))  # 記号を空白に
    words = text.translate(table).split()  # スペースで分割
    vec = [model[word] for word in words if word in model]  # 1語ずつベクトル化

    return torch.tensor(sum(vec) / len(vec))  # 平均ベクトルをTensor型に変換して出力


with open("word2vec.pkl", "rb") as f:
    model = pickle.load(f)

train = pd.read_csv("train.txt", sep='\t')
valid = pd.read_csv("valid.txt", sep='\t')
test = pd.read_csv("test.txt", sep='\t')

X_train = torch.stack([trans_word2vec(text) for text in train['TITLE']])
X_valid = torch.stack([trans_word2vec(text) for text in valid['TITLE']])
X_test = torch.stack([trans_word2vec(text) for text in test['TITLE']])

category = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category[x]).values)

torch.save(X_train, 'X_train.pt')
torch.save(X_valid, 'X_valid.pt')
torch.save(X_test, 'X_test.pt')
torch.save(y_train, 'y_train.pt')
torch.save(y_valid, 'y_valid.pt')
torch.save(y_test, 'y_test.pt')


# print(X_train.size())
# print(X_train)

# print(y_train.size())
# print(y_train)

"""
torch.Size([10684, 300])
tensor([[ 0.0208,  0.0613, -0.0656,  ..., -0.0120,  0.0970, -0.0495],
        [-0.0084,  0.0344, -0.0283,  ..., -0.2322,  0.0277,  0.0302],
        [ 0.0435, -0.0432,  0.0039,  ..., -0.0295, -0.0057, -0.0554],
        ...,
        [-0.0894,  0.0722, -0.0035,  ..., -0.0381,  0.0069,  0.0325],
        [ 0.0809,  0.0444,  0.0629,  ...,  0.0885,  0.0944,  0.0060],
        [ 0.1130,  0.1005, -0.1022,  ...,  0.0842, -0.0102, -0.0234]])
torch.Size([10684])
tensor([2, 2, 1,  ..., 0, 0, 0])
"""
