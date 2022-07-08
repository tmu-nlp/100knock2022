""'''
81. RNNによる予測
RNNを用い，単語列xからカテゴリyを予測するモデルを実装
活性化関数gはtanh/ReLU
d_w,d_h = 300, 50
'''
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from knock80 import make_ids4words
import string

# load data
train = pd.read_csv('../chapter06/train_re.txt', sep='\t', names=['CATEGORY', 'TITLE'])
valid = pd.read_csv('../chapter06/valid_re.txt', sep='\t', names=['CATEGORY', 'TITLE'])
test = pd.read_csv('../chapter06/test_re.txt', sep='\t', names=['CATEGORY', 'TITLE'])
word2id = make_ids4words(train)

def get_ids(text, word2id=word2id, unk=0):
    # 記号をスペースに置換,スペースで分割したID列に変換(辞書になければunkで0を返す)
    res = []
    table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    for word in text.translate(table).split():
        res.append(word2id.get(word, unk))
    return res

# define RNN model
class RNN(nn.Module):
    def __init__(self, hidden_size, emb_size, vocab_size, padding_idx, output_size, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    # forward learning
    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        x = self.emb(x)
        y, hidden = self.rnn(x, hidden)
        y = self.fc(y[:, -1, :])
        return y

class MyDataset(Dataset):
    def __init__(self, X, y, get_ids):
        self.X = X
        self.y = y
        self.get_ids = get_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        text = self.X[index]
        inputs = self.get_ids(text)
        dic = {}
        dic['inputs'] = torch.tensor(inputs, dtype=torch.int64)  # store id features as tensor
        dic['labels'] = torch.tensor(self.y[index], dtype=torch.int64)
        return dic


# make labels
category = {'b':0, 't':1, 'e':2, 'm':3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category[x]).values)

# make dataset
dataset_train = MyDataset(train['TITLE'], y_train, get_ids)
dataset_valid = MyDataset(valid['TITLE'], y_valid, get_ids)
dataset_test = MyDataset(test['TITLE'], y_test, get_ids)

# d_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)

if __name__ == '__main__':
    print(f'len(Dataset):{len(dataset_train)}')
    print('index in Dataset:')
    for v in dataset_train[1]:
        print(f'{v} : {dataset_train[1][v]}')   # id(freq) features

    # set parameter
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    model = RNN(emb_size=EMB_SIZE, hidden_size=HIDDEN_SIZE,vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=OUTPUT_SIZE, device=None)
    for i in range(10):
        X = dataset_train[i]['inputs']
        print(X)
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))
        # for data in d_loader_train:
        #     x = data['inputs']
        #     print(x.size())






