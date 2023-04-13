import torch
from torch import nn
from torch.utils.data import Dataset
from knock80 import *


def get_id(sentence, word_id):
    words = sentence.split()
    ids = [word_id[word.lower()] for word in words]
    return ids


class CreateDataset(Dataset):
    def __init__(self, X, y, get_id):
        self.X = X
        self.y = y
        self.get_id = get_id

    def __len__(self):
        '''len(Dataset)で返す値を指定'''
        return len(self.y)

    def __getitem__(self, index):
        '''Dataset[index]で返す値を指定'''
        item = {}
        text = self.X[index]
        inputs = self.get_id(text, word_id)
        item['inputs'] = torch.tensor(inputs, dtype=torch.int64)
        item['labels'] = torch.tensor(self.y[index], dtype=torch.int64)
        return item


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size,
                          nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()
        emb = self.emb(x)
        # (1, 300, 単語数)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden


# Darasetの作成
dataset_train = CreateDataset(train_data, y_train, tokenizer)
dataset_valid = CreateDataset(valid_data, y_valid, tokenizer)
dataset_test = CreateDataset(test_data, y_test, tokenizer)

# パラメータの設定
VOCAB_SIZE = len(set(word_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

# モデルの定義
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

# 先頭10件の予測値取得
for i in range(10):
    X = dataset_train[i]['inputs']
    print(torch.softmax(model(X.unsqueeze(0)), dim=-1))
