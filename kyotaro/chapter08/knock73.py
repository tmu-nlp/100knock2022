from cProfile import label
from pickletools import optimize
from re import A, X
from numpy import reshape
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from knock71 import NeuralNetwork
import pickle

class NewsDataset(Dataset):
    """Datasetクラスの作成"""
    def __init__(self, features, label):
        """オブジェクトの初期化"""
        self.X = features
        self.y = label
    
    def __len__(self):
        """データセットのサンプル数を返す"""
        return len(self.y)
    
    def __getitem__(self, idx):
        """idxに対応するサンプルを返す"""
        return [self.X[idx], self.y[idx]]


def calculate_SGD(model, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        loss_train = 0.0
        for batch, (X, y) in enumerate(train_dataloader):
            # 予測と損失
            pred = model(X)
            loss = loss_fn(pred, y)

            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 損失を足し合わせていく
            loss_train += loss.item()

        # バッチで平均をとる
        loss_train = loss_train / batch

        model.eval()
        with torch.no_grad():  # 勾配計算を不要にする
            inputs, labels = next(iter(valid_dataloader))
            outputs = model(inputs)
            loss_valid = loss_fn(outputs, labels) 
        torch.save(model, "model.pt")
        
        # print(f'epoch : {epoch + 1}, loss_train : {loss_train}, loss_valid : {loss_valid}')


# 訓練データのDatasetを作成
X_train = torch.load("tensor/X_train.pt")
Y_train = torch.load("tensor/Y_train.pt")
train_dataset = NewsDataset(X_train, Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# 開発データのDatasetを作成
X_valid = torch.load("tensor/X_valid.pt")
Y_valid = torch.load("tensor/Y_valid.pt")
valid_dataset = NewsDataset(X_valid, Y_valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True)

# テストデータのDatasetを作成
X_test = torch.load("tensor/X_test.pt")
Y_test = torch.load("tensor/Y_test.pt")
test_dataset = NewsDataset(X_test, Y_test)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


# モデル
model = NeuralNetwork(300, 4)
# 損失関数
loss_fn = nn.CrossEntropyLoss()
# 最適化器
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

calculate_SGD(model, loss_fn, optimizer, 100)

"""
epoch : 1, loss_train : 1.3762803499617602, loss_valid : 1.198346734046936
epoch : 2, loss_train : 1.0784587908728902, loss_valid : 1.0155969858169556
epoch : 3, loss_train : 0.9426089124849638, loss_valid : 0.9135593175888062
epoch : 4, loss_train : 0.8602391309259994, loss_valid : 0.8456618785858154
epoch : 5, loss_train : 0.8024395407580418, loss_valid : 0.7956520915031433
epoch : 6, loss_train : 0.758113515737403, loss_valid : 0.7560906410217285
epoch : 7, loss_train : 0.7222616576159963, loss_valid : 0.7235469818115234
epoch : 8, loss_train : 0.6922502766702925, loss_valid : 0.6960760354995728
epoch : 9, loss_train : 0.6665192338342338, loss_valid : 0.6720907092094421
epoch : 10, loss_train : 0.6438780142271601, loss_valid : 0.6513492465019226
"""

# https://qiita.com/tatsuya11bbs/items/86141fe3ca35bdae7338