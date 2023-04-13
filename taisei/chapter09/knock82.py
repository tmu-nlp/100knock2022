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

        torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'./output/knock82/checkpoint{i + 1}.pt')
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

def plot_log(log):
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
    plt.savefig("./output/knock82_loss_acc.png")


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
    BATCH_SIZE = 1

    model = RNN(VOCAB_SIZE, PADDING_IDX, OUTPUT_SIZE, EMB_SIZE, HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    log = train_model(dataset_train, dataset_valid, model, criterion, optimizer, BATCH_SIZE, 10)
    plot_log(log)
    _, acc_train = calculate_loss_acc(model, dataset_train)
    _, acc_valid = calculate_loss_acc(model, dataset_valid)
    print('正解率')
    print(f'訓練データ：{acc_train:.5f}')
    print(f'開発データ：{acc_valid:.5f}')

"""
epoch: 1, loss_train: 1.0743, accuracy_train: 0.5626, loss_valid: 1.1114, accuracy_valid: 0.5232, 54.8663sec
 20%|████████████████▍                                                                 | 2/10 [01:51<07:27, 55.90s/it]
epoch: 3, loss_train: 0.9247, accuracy_train: 0.6516, loss_valid: 0.9969, accuracy_valid: 0.6207, 56.3602sec
 10%|████████▏                                                                         | 1/10 [00:54<08:13, 54.87s/it]
epoch: 2, loss_train: 1.0083, accuracy_train: 0.6018, loss_valid: 1.0663, accuracy_valid: 0.5727, 56.6219sec
 30%|████████████████████████▌                                                         | 3/10 [02:47<06:32, 56.11s/it]
epoch: 4, loss_train: 0.8190, accuracy_train: 0.7022, loss_valid: 0.9318, accuracy_valid: 0.6619, 56.1121sec
 40%|████████████████████████████████▊                                                 | 4/10 [03:43<05:36, 56.11s/it]
epoch: 5, loss_train: 0.7172, accuracy_train: 0.7447, loss_valid: 0.8214, accuracy_valid: 0.7091, 55.0210sec
 50%|█████████████████████████████████████████                                         | 5/10 [04:38<04:38, 55.72s/it]
epoch: 6, loss_train: 0.6686, accuracy_train: 0.7588, loss_valid: 0.8358, accuracy_valid: 0.7016, 56.6307sec
 60%|█████████████████████████████████████████████████▏                                | 6/10 [05:35<03:44, 56.03s/it]
epoch: 7, loss_train: 0.6115, accuracy_train: 0.7800, loss_valid: 0.7661, accuracy_valid: 0.7279, 56.3789sec
 70%|█████████████████████████████████████████████████████████▍                        | 7/10 [06:31<02:48, 56.14s/it]
epoch: 8, loss_train: 0.5749, accuracy_train: 0.7954, loss_valid: 0.7473, accuracy_valid: 0.7286, 56.0391sec
 80%|█████████████████████████████████████████████████████████████████▌                | 8/10 [07:28<01:52, 56.11s/it]
epoch: 9, loss_train: 0.5608, accuracy_train: 0.7999, loss_valid: 0.7434, accuracy_valid: 0.7286, 56.5517sec
 90%|█████████████████████████████████████████████████████████████████████████▊        | 9/10 [08:24<00:56, 56.25s/it]
epoch: 10, loss_train: 0.5557, accuracy_train: 0.7990, loss_valid: 0.7395, accuracy_valid: 0.7354, 55.9337sec
100%|█████████████████████████████████████████████████████████████████████████████████| 10/10 [09:20<00:00, 56.05s/it]
正解率
訓練データ：0.79901
開発データ：0.73538
"""