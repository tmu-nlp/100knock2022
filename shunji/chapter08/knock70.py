import string
import torch
import pickle
import pandas as pd


def transform_w2v(text):
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    words = text.translate(table).split()  # 記号をスペースに置換し，スペースで分割してリスト化
    vec = [
        model[word] for word in words if word in model
    ]  # 1語ずつベクトル化(type: list of ndarray)

    return torch.from_numpy(sum(vec) / len(vec))  # 平均ベクトルをTensor型で返す


# モデルと学習，検証，評価データの読み込み
model = pickle.load(open("model.pkl", "rb"))
train = pd.read_csv("train.txt", sep="\t")
valid = pd.read_csv("valid.txt", sep="\t")
test = pd.read_csv("test.txt", sep="\t")

# 特徴ベクトルの作成
# 次元数：300*1 -> データ数*300*1
X_train = torch.stack([transform_w2v(text) for text in train["TITLE"]])
X_valid = torch.stack([transform_w2v(text) for text in valid["TITLE"]])
X_test = torch.stack([transform_w2v(text) for text in test["TITLE"]])

# 確認用
# print(X_train.size())
# print(X_train)

# ラベルベクトルの作成
category_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
y_train = torch.tensor(train["CATEGORY"].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid["CATEGORY"].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test["CATEGORY"].map(lambda x: category_dict[x]).values)

# 確認用
# print(y_train.size())
# print(y_train)

# 保存
torch.save(X_train, "X_train.pt")
torch.save(X_valid, "X_valid.pt")
torch.save(X_test, "X_test.pt")
torch.save(y_train, "y_train.pt")
torch.save(y_valid, "y_valid.pt")
torch.save(y_test, "y_test.pt")
