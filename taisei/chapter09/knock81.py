import torch 
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

class RNN(nn.Module):
    def __init__(self, vocab_size, padding_idx, output_size, emb_size=300, hidden_size=50): #ネットワークが構成するレイヤー
        torch.manual_seed(0)
        super().__init__()
        self.hidden_size = hidden_size
        #単語IDをone-hotベクトルに変換する
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) #全結合？

    def forward(self, input):
        self.batch_size = input.size()[0]
        hidden = self.init_hidden() #h0のゼロベクトル
        emb = self.emb(input)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden

    
class MakeDataset(Dataset):
    def __init__(self, X, y, get_ids):
        self.X = X
        self.y = y
        self.get_ids = get_ids

    def __len__(self):  # len()でサイズを返す
        return len(self.y)

    def __getitem__(self, index):  # getitem(index)で指定インデックスのベクトルとlabエルを返す
        text = self.X[index]
        inputs = self.get_ids(text, word2id)

        return {
        'inputs': torch.tensor(inputs, dtype=torch.int64),
        'labels': torch.tensor(self.y[index], dtype=torch.int64)
        }

def make_ids(train_data):

    count_dict = defaultdict(lambda: 0)

    for line in train_data['TITLE']:
        words = line.strip().split()
        for word in words:
            count_dict[word] += 1
    count_dict = sorted(count_dict.items(), key=lambda x:x[1], reverse=True)

    word2id = defaultdict(int)
    for i, (word, cnt) in enumerate(count_dict):
        if cnt <= 1:
            break
        word2id[word] = i + 1
    return word2id

def get_ids(text, word2id):
    words = text.strip().split()
    ids = []
    for word in words:
        ids.append(word2id[word])
    return ids


if __name__ == "__main__":
    train_data = pd.read_csv('../chapter06/output/train.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    valid_data = pd.read_csv('../chapter06/output/valid.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    test_data = pd.read_csv('../chapter06/output/test.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    word2id = make_ids(train_data)
    
    Y_train = train_data["CATEGORY"]
    X_train_text = train_data["TITLE"]
    Y_valid = valid_data["CATEGORY"]
    X_valid_text = valid_data["TITLE"]
    Y_test = test_data["CATEGORY"]
    X_test_text = test_data["TITLE"]

    Y_train = Y_train.map({"b":0, "t":1, "e":2, "m":3})
    Y_valid = Y_valid.map({"b":0, "t":1, "e":2, "m":3})
    Y_test = Y_test.map({"b":0, "t":1, "e":2, "m":3})

    dataset_train = MakeDataset(X_train_text, Y_train, get_ids)
    dataset_valid = MakeDataset(X_valid_text, Y_valid, get_ids)
    dataset_test = MakeDataset(X_test_text, Y_test, get_ids)

    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    model = RNN(VOCAB_SIZE, PADDING_IDX, OUTPUT_SIZE, EMB_SIZE, HIDDEN_SIZE)

    for i in range(10):
        X = dataset_train[i]['inputs']
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))

"""
tensor([[0.3730, 0.1328, 0.2957, 0.1985]], grad_fn=<SoftmaxBackward0>)
tensor([[0.1928, 0.2733, 0.3081, 0.2258]], grad_fn=<SoftmaxBackward0>)
tensor([[0.4477, 0.1702, 0.1207, 0.2614]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2927, 0.2172, 0.2848, 0.2052]], grad_fn=<SoftmaxBackward0>)
tensor([[0.3822, 0.2186, 0.0995, 0.2997]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2246, 0.2262, 0.3556, 0.1936]], grad_fn=<SoftmaxBackward0>)
tensor([[0.3340, 0.1933, 0.2389, 0.2338]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2229, 0.3684, 0.2145, 0.1942]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2569, 0.2575, 0.2721, 0.2134]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2657, 0.3404, 0.1250, 0.2688]], grad_fn=<SoftmaxBackward0>)
"""