import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

def tokenizer(text, voc_id, unk=0):
    ids = []
    for word in text.split():
        ids.append(voc_id.get(word, unk))
    return ids

class RNN(nn.Module):
    """RNNモデルの構築"""
    def __init__(self, hidden_size, vocab_size, emb_size, padding_idx, output_size):
        super().__init__()
        self.hidden_size = hidden_size  # 隠れ層の数
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx)  # 埋め込み層（one-hotベクトルを返すような関数）
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)  # RNNの隠れ層
        self.fc = nn.Linear(hidden_size, output_size)  # 出力層
    
    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()  # 最初の隠れ層
        emb = self.emb(x)  # 入力を埋め込み
        out, hidden = self.rnn(emb, hidden)  # RNNなので前の隠れ層を入力に入れる
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden


class CreateDataset(Dataset):
    """入力とラベルを受け取り、toknizeしてからTensorに"""
    def __init__(self, X, y, tokenizer, voc_id):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.voc_id = voc_id
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        text = self.X[index]  # index番目の文を選択
        inputs = self.tokenizer(text, self.voc_id)  # 選択した文をIDに変換
        items = {}
        items['inputs'] = torch.tensor(inputs, dtype=torch.int64)  # ID列
        items['labels'] = torch.tensor(self.y[index], dtype=torch.int64)  # ラベル
        return items


if __name__ == "__main__":
    train = pd.read_csv("../chapter06/train.txt", sep="\t")
    valid = pd.read_csv("../chapter06/valid.txt", sep="\t")
    test = pd.read_csv("../chapter06/test.txt", sep="\t")

    frec = defaultdict(lambda: 0)  # 単語の頻度
    voc_id = defaultdict(lambda: 0)  # 単語のID

    for line in train["TITLE"]:
        words = line.strip().split()
        for word in words:
            frec[word] += 1

    frec = sorted(frec.items(), key=lambda x: x[1], reverse=True)  # 頻度順にソート

    for i, word in enumerate(frec):
        if word[1] >= 2:
            voc_id[word[0]] = i + 1


    # データをchapter06から持ってきて、ラベルを数値に変換
    category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    y_train = pd.read_csv("../chapter06/train.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values
    y_valid = pd.read_csv("../chapter06/valid.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values
    y_test = pd.read_csv("../chapter06/test.txt", sep="\t")["CATEGORY"].map(lambda x: category_dict[x]).values

    # Datasetを作成
    dataset_train = CreateDataset(train["TITLE"], y_train, tokenizer, voc_id)
    dataset_valid = CreateDataset(valid["TITLE"], y_valid, tokenizer, voc_id)
    dataset_test = CreateDataset(test["TITLE"], y_test, tokenizer, voc_id)

    # パラメータ設定
    VOCAB_SIZE = len(set(voc_id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(voc_id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    # モデル
    model = RNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE)
    
    # 予測
    # for i in range(10):
    #     X = dataset_train[i]['inputs']
    #     print(torch.softmax(model(X.unsqueeze(0)), dim=-1))