from time import time  # 時間の計測
import japanize_matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn, optim
import torch
from torch.nn import functional


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


class MultiLayerNet(nn.Module):
    '''多層ニューラルネットワークの定義
    入力層, 中間層, (バッチ正規化), 出力層'''

    def __init__(self, input_size, mid_size, output_size, mid_layers):
        super().__init__()
        self.mid_layers = mid_layers
        self.fc = nn.Linear(input_size, mid_size)
        self.fc_mid = nn.Linear(mid_size, mid_size)
        self.fc_out = nn.Linear(mid_size, output_size)
        self.bn = nn.BatchNorm1d(mid_size)

    def forward(self, x):
        x = functional.relu(self.fc(x))  # ReLU関数により活性化
        for _ in range(self.mid_layers):
            x = functional.relu(self.bn(self.fc_mid(x)))
        x = functional.relu(self.fc_out(x))
        return x


def calclate_acuracy_cuda(model, loader, device):
    '''正解率の計算, cuda指定の引数を追加'''
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    accuracy = correct / total
    return accuracy


def calculate_loss_cuda(model, criterion, loader, device):
    '''損失の計算'''
    model.eval()
    loss = 0.
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss += criterion(outputs, labels).item()

    return loss / len(loader)


def train_model_cuda(train_dataset, valid_dataset, batch_size, model, criterion, optimizer, num_epochs, device=None):
    '''knock76までに書いたデータローダの作成(バッチサイズを変えるため)とモデルの学習プロセスを関数化
    バッチサイズを引数で指定して学習する'''

    model.to(device)

    # DataLoaderの作成 (DataLoaderはDatasetからバッチごとにサンプルを取得して、ミニバッチを作成する)
    # Dataset = [全データ] --> DataLoader = [[バッチ1], [バッチ2], ... [バッチn]]
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=True)

    # スケジューラ 学習率を自動で調節する(例えば学習が進むに従い、学習率を下げるなど)
    scheduler = optim.lr_scheduler.StepLR(optimizer, num_epochs, gamma=0.5)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

    # 学習
    trains = []
    valids = []
    for epoch in range(num_epochs):

        begin = time()  # 計測開始時間

        model.train()  # 訓練モード (Moduleは training/evaluation の2種類のモードを持っている)

        # 順伝播、逆伝播、重み更新
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 計算した勾配の初期化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 誤差逆伝播
            optimizer.step()  # 重み更新

        # 評価
        model.eval()

        # 訓練データの損失、正解率を計算
        loss_train = calculate_loss_cuda(
            model, criterion, train_dataloader, device)
        acc_train = calclate_acuracy_cuda(model, train_dataloader, device)

        # 検証データの損失、正解率を計算
        loss_valid = calculate_loss_cuda(
            model, criterion, valid_dataloader, device)
        acc_valid = calclate_acuracy_cuda(model, valid_dataloader, device)

        trains.append([loss_train, acc_train])
        valids.append([loss_valid, acc_valid])

        end = time()  # 計測終了時間
        log_time = end - begin  # 実行時間

        # 可視化
        print('-'*20, f'batch size: {batch_size}', '-'*20)
        print(f'epoch: {epoch}')
        print(f'loss(train):{loss_train}\taccuracy(train):{acc_train}')
        print(f'loss(valid):{loss_valid}\taccuracy(valid):{acc_valid}')
        print(f'{end - begin}[sec]')

        # 検証データの損失が3エポック連続で減少しなかった場合は学習を終了する
        if epoch > 2 and valids[epoch - 3][0] <= valids[epoch - 2][0] <= valids[epoch - 1][0] <= valids[epoch][0]:
            break

        scheduler.step()  # スケジューラを1ステップ進める

    return [trains, valids]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データの読み込み
X_train, y_train = torch.load('./X_train.pt'), torch.load('./y_train.pt')
X_valid, y_valid = torch.load('./X_valid.pt'), torch.load('./y_valid.pt')
X_test, y_test = torch.load('./X_test.pt'), torch.load('./y_test.pt')

# Datasetの作成
train_dataset = dataset(X_train, y_train)
valid_dataset = dataset(X_valid, y_valid)
test_dataset = dataset(X_test, y_test)

# モデル定義
# モデル定義 (input_size, mid_size, output_size, mid_layers)
model = MultiLayerNet(300, 200, 4, 1)
criterion = nn.CrossEntropyLoss()  # 損失関数の定義
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 最適化手法の定義

# 学習 (train_dataset, valid_dataset, batch_size, model, criterion, optimizer, num_epochs, device=None)
log = train_model_cuda(train_dataset, valid_dataset, 64,
                       model, criterion, optimizer, 1000, device)

# 損失の訓練状況の描画
plt.plot(np.array(log[0]).T[0], label='train')
plt.plot(np.array(log[1]).T[0], label='valid')
plt.xlabel('学習回数')
plt.ylabel('損失')
plt.legend()
# plt.savefig('./results/output79_loss.png')
plt.show()

# 正解率の訓練状況の描画
plt.plot(np.array(log[0]).T[1], label='train')
plt.plot(np.array(log[1]).T[1], label='valid')
plt.xlabel('学習回数')
plt.ylabel('正解率')
plt.legend()
# plt.savefig('./results/output79_accuracy.png')
plt.show()
