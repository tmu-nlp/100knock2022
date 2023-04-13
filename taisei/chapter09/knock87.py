import torch 
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from gensim.models import KeyedVectors
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        #x:(batch_size, 文長)
        emb = self.emb(x).unsqueeze(1)
        #emb:[batch_size, in_channels, 文長, emb_size]
        conv = self.conv(emb)
        #conv:[batch_size, out_channels, 文長, 1]
        act = F.relu(conv.squeeze(3))
        #act→[batch_size, out_channels, 文長]
        max_pool = F.max_pool1d(act, act.size()[2])
        out = self.fc(self.drop(max_pool.squeeze(2)))
        return out


class RNN(nn.Module): #初期値を引数に追加する
    def __init__(self, vocab_size, padding_idx, output_size, emb_size=300, hidden_size=50, num_layers=1, emb_weights=None, bidirectional=False): #ネットワークが構成するレイヤー
        torch.manual_seed(0)
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1 #単方向なら1、双方向なら2
        #単語IDをone-hotベクトルに変換する
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity='tanh', bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size) #全結合？
        

    def forward(self, input):
        self.batch_size = input.size()[0]
        hidden = self.init_hidden() #h0のゼロベクトル
        emb = self.emb(input)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        return hidden

class Padsequence():
    # バッチ内での単語長は同じ出なければならないので長井やつに合わせてパディングする
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x["inputs"].shape[0], reverse=True)
        sequences = [x["inputs"] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x["labels"] for x in sorted_batch])

        return {"inputs": sequences_padded, "labels": labels}

    
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

def calculate_loss_acc(model, dataset, device=None, criterion=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0
    total = 0
    corr = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)

            outputs = model(inputs) # 順伝播
            if criterion != None: # 損失計算
                loss += criterion(outputs, labels).item()
            # 正解率
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            corr += (pred == labels).sum().item()
    return loss/len(dataset), corr/total


def train_model(dataset_train, dataset_valid, model, criterion, optimizer, batch_size=1, epochs=10, collate_fn=None, device=None):
    model.to(device)

    # dataloaderを作成する
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    #スケジューラの設定　学習率をcosine関数に従って初期値からeta_minまで小さくする
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5, last_epoch=-1)

    log_train = []
    log_valid = []
    for i in tqdm(range(epochs)):
        s_time = time.time()

        model.train() #訓練モード
        total_loss = 0
        for data in dataloader_train:
            optimizer.zero_grad()
            inputs = data["inputs"].to(device)
            labels = data['labels'].to(device)

            outputs = model(inputs) #順伝藩
            loss = criterion(outputs, labels) 
            loss.backward() #逆伝藩
            optimizer.step() #重み更新

        model.eval() #評価モード

        #損失と正解率の算出
        loss_train, acc_train = calculate_loss_acc(model, dataset_train, device, criterion=criterion)
        loss_valid, acc_valid = calculate_loss_acc(model, dataset_valid, device, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./output/knock87/checkpoint{i + 1}.pt')
        e_time = time.time()

        print(f'epoch: {i + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec')

        #検証データのロスが3エポック連続で低下しなかったら学習終了
        if i > 2 and log_valid[i-3][0] <= log_valid[i-2][0] <= log_valid[i-1][0] <= log_valid[i][0]:
            break

        scheduler.step()

    return {'train': log_train, 'valid': log_valid}


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

def plot_log(log, outpath):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log['train']).T[0], label='train')
    ax[0].plot(np.array(log['valid']).T[0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log['train']).T[1], label='train')
    ax[1].plot(np.array(log['valid']).T[1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.savefig(outpath)

def word2vec(word2id):
    model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin.gz", binary=True)
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
    words_in_pretrained = 0
    for i, word in enumerate(word2id.keys()):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
    weights = torch.from_numpy(weights.astype((np.float32)))
    return weights

if __name__ == "__main__":
    train_data = pd.read_csv('../chapter06/output/train.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    valid_data = pd.read_csv('../chapter06/output/valid.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    test_data = pd.read_csv('../chapter06/output/test.txt', sep='\t', names=('CATEGORY', 'TITLE'))
    word2id = make_ids(train_data)
    weights = word2vec(word2id)

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

        
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    LEARNING_RATE = 0.05
    BATCH_SIZE = 32
    NUM_EPOCH = 10

    # モデルの定義
    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)
    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()
    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # モデルの学習
    log = train_model(dataset_train, dataset_valid, model, criterion, optimizer, BATCH_SIZE, NUM_EPOCH, collate_fn=Padsequence(PADDING_IDX))
   
    plot_log(log, "./output/knock87_loss_acc.png")
    _, acc_train = calculate_loss_acc(model, dataset_train)
    _, acc_valid = calculate_loss_acc(model, dataset_valid)
    print('正解率')
    print(f'訓練データ：{acc_train:.5f}')
    print(f'開発データ：{acc_valid:.5f}')
    
"""
epoch: 1, loss_train: 1.0141, accuracy_train: 0.6183, loss_valid: 1.0339, accuracy_valid: 0.5997, 10.8229sec
 10%|████████▏                                                                         | 1/10 [00:10<01:37, 10.82s/it]
epoch: 2, loss_train: 0.8676, accuracy_train: 0.7013, loss_valid: 0.9224, accuracy_valid: 0.6627, 10.6383sec
 20%|████████████████▍                                                                 | 2/10 [00:21<01:25, 10.71s/it]
epoch: 3, loss_train: 0.7613, accuracy_train: 0.7382, loss_valid: 0.8472, accuracy_valid: 0.6882, 10.6728sec
 30%|████████████████████████▌                                                         | 3/10 [00:32<01:14, 10.70s/it]
epoch: 4, loss_train: 0.6617, accuracy_train: 0.7721, loss_valid: 0.7786, accuracy_valid: 0.7106, 10.6699sec
 40%|████████████████████████████████▊                                                 | 4/10 [00:42<01:04, 10.69s/it]
epoch: 5, loss_train: 0.5940, accuracy_train: 0.7924, loss_valid: 0.7414, accuracy_valid: 0.7294, 10.7567sec
 50%|█████████████████████████████████████████                                         | 5/10 [00:53<00:53, 10.71s/it]
epoch: 6, loss_train: 0.5477, accuracy_train: 0.8065, loss_valid: 0.7176, accuracy_valid: 0.7384, 11.1776sec
 60%|█████████████████████████████████████████████████▏                                | 6/10 [01:04<00:43, 10.87s/it]
epoch: 7, loss_train: 0.5122, accuracy_train: 0.8223, loss_valid: 0.7023, accuracy_valid: 0.7466, 10.9315sec
 70%|█████████████████████████████████████████████████████████▍                        | 7/10 [01:15<00:32, 10.89s/it]
epoch: 8, loss_train: 0.4925, accuracy_train: 0.8285, loss_valid: 0.6920, accuracy_valid: 0.7496, 10.6903sec
 80%|█████████████████████████████████████████████████████████████████▌                | 8/10 [01:26<00:21, 10.83s/it]
epoch: 9, loss_train: 0.4832, accuracy_train: 0.8332, loss_valid: 0.6870, accuracy_valid: 0.7519, 10.6705sec
 90%|█████████████████████████████████████████████████████████████████████████▊        | 9/10 [01:37<00:10, 10.78s/it]
epoch: 10, loss_train: 0.4808, accuracy_train: 0.8351, loss_valid: 0.6861, accuracy_valid: 0.7534, 10.6621sec
100%|█████████████████████████████████████████████████████████████████████████████████| 10/10 [01:47<00:00, 10.77s/it]
正解率
訓練データ：0.83508
開発データ：0.75337
"""