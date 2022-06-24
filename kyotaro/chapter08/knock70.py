from email import header
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import string
import torch


def transform_w2v(text):
    table = str.maketrans(string.punctuation, ' ' *
                          len(string.punctuation))  # 句読点のところに空白を入れるようなテーブル
    words = text.translate(table).split()  # tableに従い変換した後にスペースで分割
    vec = [model[word] for word in words if word in model]  # ベクトル化
    return torch.tensor(sum(vec) / len(vec))  # x_i

# 問題50で作成したデータ
train_file = pd.read_table("../chapter06/train.txt")
valid_file = pd.read_table("../chapter06/valid.txt")
test_file = pd.read_table("../chapter06/test.txt")

# 問題60で作成した単語ベクトル
model_file = "../chapter07/model.sav"
with open(model_file, "rb") as data:
    model = pickle.load(data)

# それぞれのデータに対してベクトル化、x_iをどんどんくっつけていく
X_train = torch.stack([transform_w2v(text) for text in train_file["TITLE"]])
X_valid = torch.stack([transform_w2v(text) for text in valid_file["TITLE"]])
X_test = torch.stack([transform_w2v(text) for text in test_file["TITLE"]])

# ラベルベクトル
label = {'b':0, 't':1, 'e':2, 'm':3}
Y_train = torch.tensor(train_file["CATEGORY"].map(lambda x: label[x]).values)
Y_valid = torch.tensor(valid_file["CATEGORY"].map(lambda x: label[x]).values)
Y_test = torch.tensor(test_file["CATEGORY"].map(lambda x: label[x]).values)

torch.save(X_train, 'tensor/X_train.pt')
torch.save(Y_train, 'tensor/Y_train.pt')
torch.save(X_valid, 'tensor/X_valid.pt')
torch.save(Y_valid, 'tensor/Y_valid.pt')
torch.save(X_test, 'tensor/X_test.pt')
torch.save(Y_test, 'tensor/Y_test.pt')