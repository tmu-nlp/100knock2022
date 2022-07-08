import time
import torch
import string
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# データの読込、抽出、分割
df = pd.read_csv('./100knock2022/DUAN/chapter06/newsCorpora.csv', header = None, sep = '\t', names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size = 0.2, shuffle = True, random_state = 123, stratify = df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size = 0.5, shuffle = True, random_state = 123, stratify = valid_test['CATEGORY'])

# インデックスを連番に振り直す
train.reset_index(drop = True, inplace = True)
valid.reset_index(drop = True, inplace = True)
test.reset_index(drop = True, inplace = True)

# 単語の頻度を計算する
d = defaultdict(int)
table = str.maketrans(string.punctuation, ' '*len(string.punctuation))  
for text in train['TITLE']:
    for word in text.translate(table).split():
        d[word] += 1
d = sorted(d.items(), key=lambda x:x[1], reverse=True)

# 単語の辞書を作成する
word2id = {word: i + 1 for i, (word, cnt) in enumerate(d) if cnt > 1} 

# 単語列をID番号の列に変換する
def tokenizer(text, word2id = word2id, unk = 0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [word2id.get(word, unk) for word in text.translate(table).split()]

class RNN(nn.Module): # ニューラルネットワークモジュールの定義
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx = padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity = 'tanh', batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): # インスタンスを順次呼び出す
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()  
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self): # 隠し状態の初期化を行う
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)
        return hidden

class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer): # 構成要素を指定する
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self): # len(Dataset)で返す値を指定する
        return len(self.y)

    def __getitem__(self, index): # Dataset[index]で返す値を指定する
        text = self.X[index]
        inputs = self.tokenizer(text)
        return {'inputs': torch.tensor(inputs, dtype = torch.int64),'labels': torch.tensor(self.y[index], dtype = torch.int64)}

# ラベルベクトルを作成する
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values

# Datasetを作成する
dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer)
dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer)
dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer)

# 損失と正解率を計算する
def calculate_loss_and_accuracy(model, dataset, device = None, criterion = None):
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device) # デバイスを指定する
            labels = data['labels'].to(device)
            outputs = model(inputs) # 順伝播

            if criterion != None: # 損失計算
                loss += criterion(outputs, labels).item()
            
            # 正解率計算
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss / len(dataset), correct / total

def visualize_logs(log): # ログを可視化するための関数を定義する
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    ax[0].plot(np.array(log['train']).T[0], label = 'train')
    ax[0].plot(np.array(log['valid']).T[0], label = 'valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log['train']).T[1], label = 'train')
    ax[1].plot(np.array(log['valid']).T[1], label = 'valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.show()

# モデルの学習を実行する
def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn = None, device = None):
    model.to(device)
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size = 1, shuffle = False)
    
    # スケジューラを設定する
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min = 1e-5, last_epoch = -1)
    
    # 学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        s_time = time.time() # 開始時刻を記録する
        model.train() # 訓練モードに設定する
        for data in dataloader_train:
            optimizer.zero_grad() # 勾配をゼロで初期化する
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            
            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval() # 評価モードに設定する

        # 損失と正解率を計算する
        loss_train, acc_train = calculate_loss_and_accuracy(model, dataset_train, device, criterion = criterion)
        loss_valid, acc_valid = calculate_loss_and_accuracy(model, dataset_valid, device, criterion = criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])
        
        # チェックポイントを保存する
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
        e_time = time.time() # 終了時刻を記録する
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec') 
        
        # 検証データの損失が3エポック連続で低下しなかったら学習終了
        if epoch > 2 and log_valid[epoch - 3][0] <= log_valid[epoch - 2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]:
            break
        scheduler.step() # スケジューラを1ステップ進める
    return {'train': log_train, 'valid': log_valid}

# パラメータを設定する
VOCAB_SIZE = len(set(word2id.values())) + 1 
EMB_SIZE = 300
PADDING_IDX = len(set(word2id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = 10

# モデル、損失関数、オプティマイザを定義する
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# モデルを学習する
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)
visualize_logs(log) # 可視化
n, acc_train = calculate_loss_and_accuracy(model, dataset_train)
t, acc_test = calculate_loss_and_accuracy(model, dataset_test)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')

'''
epoch: 1, loss_train: 1.0958, accuracy_train: 0.5251, loss_valid: 1.1324, accuracy_valid: 0.5000, 90.8019sec
epoch: 2, loss_train: 0.9916, accuracy_train: 0.6083, loss_valid: 1.0633, accuracy_valid: 0.5780, 100.2691sec
epoch: 3, loss_train: 0.7989, accuracy_train: 0.7092, loss_valid: 0.8704, accuracy_valid: 0.6897, 104.5324sec
epoch: 4, loss_train: 0.6759, accuracy_train: 0.7589, loss_valid: 0.8074, accuracy_valid: 0.7226, 97.9374sec
epoch: 5, loss_train: 0.6090, accuracy_train: 0.7787, loss_valid: 0.7897, accuracy_valid: 0.7264, 96.3005sec
epoch: 6, loss_train: 0.5272, accuracy_train: 0.8095, loss_valid: 0.7074, accuracy_valid: 0.7451, 110.4666sec
epoch: 7, loss_train: 0.4858, accuracy_train: 0.8239, loss_valid: 0.6772, accuracy_valid: 0.7609, 101.4913sec
epoch: 8, loss_train: 0.4557, accuracy_train: 0.8338, loss_valid: 0.6601, accuracy_valid: 0.7586, 111.3902sec
epoch: 9, loss_train: 0.4394, accuracy_train: 0.8374, loss_valid: 0.6557, accuracy_valid: 0.7661, 114.2546sec
epoch: 10, loss_train: 0.4336, accuracy_train: 0.8384, loss_valid: 0.6494, accuracy_valid: 0.7661, 100.4298sec
正解率（学習データ）：0.838
正解率（評価データ）：0.784
'''