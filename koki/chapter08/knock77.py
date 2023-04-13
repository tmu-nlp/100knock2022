import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib
from time import time  # 時間の計測


class dataset(torch.utils.data.Dataset):
    '''Datasetクラスの定義, len, getitemが必須
    データローダを使う場合や前処理定義する場合'''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return [feature, label]


class SimpleNet(nn.Module):
    '''単層ニューラルネットワークモデルの定義
    class torch.nn.ModuleはすべてのNNモジュールの基底クラス、これのサブクラスによりモデルを定義'''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size,
                            bias=False)  # Linear...全結合層(重みパラメータの作成)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    def forward(self, x):
        '''順方向の計算処理'''
        x = self.fc(x)
        return x


def calclate_acuracy(model, loader):
    '''正解率の計算'''
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    accuracy = correct / total
    return accuracy


def calculate_loss(model, criterion, loader):
    '''損失の計算'''
    model.eval()
    loss = 0.
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()

    return loss / len(loader)


def train_model(train_dataset, valid_dataset, batch_size, model, criterion, optimizer, num_epochs):
    '''knock76までに書いたデータローダの作成(バッチサイズを変えるため)とモデルの学習プロセスを関数化
    バッチサイズを引数で指定して学習する'''

    # DataLoaderの作成 (DataLoaderはDatasetからバッチごとにサンプルを取得して、ミニバッチを作成する)
    # Dataset = [全データ] --> DataLoader = [[バッチ1], [バッチ2], ... [バッチn]]
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=True)

    # 学習
    for epoch in range(num_epochs):

        begin = time()  # 計測開始時間

        model.train()  # 訓練モード (Moduleは training/evaluation の2種類のモードを持っている)

        # 順伝播、逆伝播、重み更新
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()  # 計算した勾配の初期化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 誤差逆伝播
            optimizer.step()  # 重み更新

        # 評価
        model.eval()

        # 訓練データの損失、正解率を計算
        loss_train = calculate_loss(model, criterion, train_dataloader)
        acc_train = calclate_acuracy(model, train_dataloader)

        # 検証データの損失、正解率を計算
        loss_valid = calculate_loss(model, criterion, valid_dataloader)
        acc_valid = calclate_acuracy(model, valid_dataloader)

        end = time()  # 計測終了時間

        '''
        # 可視化
        print('-'*20, f'batch size: {batch_size}', '-'*20)
        print(f'epoch: {epoch}')
        print(f'loss(train):{loss_train}\taccuracy(train):{acc_train}')
        print(f'loss(train):{loss_valid}\taccuracy(train):{acc_valid}')
        print(f'{end - begin}[sec]')
        '''
        log_time = end - begin

    return log_time


# データの読み込み
X_train, y_train = torch.load('./X_train.pt'), torch.load('./y_train.pt')
X_valid, y_valid = torch.load('./X_valid.pt'), torch.load('./y_valid.pt')
X_test, y_test = torch.load('./X_test.pt'), torch.load('./y_test.pt')

# Datasetの作成
train_dataset = dataset(X_train, y_train)
valid_dataset = dataset(X_valid, y_valid)
test_dataset = dataset(X_test, y_test)

'''
# データローダの作成
test_dataloader = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=True)
'''

# モデル定義
model = SimpleNet(300, 4)  # モデル定義
criterion = nn.CrossEntropyLoss()  # 損失関数の定義
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)  # 最適化手法の定義

# 学習
num_epochs = 20  # エポック上限
B = [2**i for i in range(num_epochs)]  # B事例数...2の累乗ごとにバッチサイズを変化

# バッチサイズを変更して学習、学習にかかる時間を保存
times = []
for batch_size in B:
    log_time = train_model(train_dataset, valid_dataset,
                           batch_size, model, criterion, optimizer, 1)
    times.append(log_time)

plt.plot(B, times)
plt.xlabel('バッチサイズ')
plt.ylabel('学習時間')
plt.xscale('log', base=2)
plt.savefig('./results/output77.png')
plt.show()
