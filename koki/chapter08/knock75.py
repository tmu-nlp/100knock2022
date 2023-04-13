import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


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


model = SimpleNet(300, 4)  # モデル定義
criterion = nn.CrossEntropyLoss()  # 損失関数の定義
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)  # 最適化手法の定義

# データの読み込み
X_train, y_train = torch.load('./X_train.pt'), torch.load('./y_train.pt')
X_valid, y_valid = torch.load('./X_valid.pt'), torch.load('./y_valid.pt')
X_test, y_test = torch.load('./X_test.pt'), torch.load('./y_test.pt')

# Datasetの作成
train_dataset = dataset(X_train, y_train)
valid_dataset = dataset(X_valid, y_valid)
test_dataset = dataset(X_test, y_test)

# Dataloaderの作成 (Datasetからサンプルを取得して、ミニバッチを作成する)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=len(valid_dataset), shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=True)


num_epochs = 100  # エポック上限
log_train = []
log_valid = []

for epoch in range(num_epochs):
    model.train()  # 訓練モード, Moduleは training/evaluation の2種類のモードを持っている。
    loss_train = 0.

    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    # 訓練データの損失、正解率を計算
    loss_train = calculate_loss(model, criterion, train_dataloader)
    acc_train = calclate_acuracy(model, train_dataloader)

    # 検証データの損失、正解率を計算
    loss_valid = calculate_loss(model, criterion, valid_dataloader)
    acc_valid = calclate_acuracy(model, valid_dataloader)

    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    # if epoch % 10 == 0:
    #print(f'epoch: {epoch}\tloss_train: {loss_train:.4f}\taccuracy_train: {acc_train:.4f}\tloss_valid: {loss_valid:.4f}\taccuracy_valid: {acc_valid:.4f}')

# 損失の訓練状況の描画
plt.plot(np.array(log_train).T[0], label='train')
plt.plot(np.array(log_valid).T[0], label='valid')
plt.xlabel('学習回数')
plt.ylabel('損失')
plt.legend()
plt.savefig('./results/output75_loss.png')
plt.show()

# 正解率の訓練状況の描画
plt.plot(np.array(log_train).T[1], label='train')
plt.plot(np.array(log_valid).T[1], label='valid')
plt.xlabel('学習回数')
plt.ylabel('正解率')
plt.legend()
plt.savefig('./results/output75_accuracy.png')
plt.show()
