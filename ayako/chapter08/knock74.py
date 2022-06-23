import torch
import numpy as np
from sklearn.metrics import accuracy_score

#学習/評価データ読み込み
x_train = np.loadtxt("./data/x_train.txt", delimiter=" ")
y_train = np.loadtxt("./data/y_train.txt")
x_test = np.loadtxt("./data/x_test.txt", delimiter=" ")
y_test = np.loadtxt("./data/y_test.txt")
#pytorchのtensor型にする
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)

#knock73で学習したモデルの読み込み
net = torch.nn.Linear(300, 4, bias=False)
net.load_state_dict(torch.load("model.pt"))

#正解率を求める
#torch.maxはtensorの(最大値, 最大値のインデックス)を返す
y_max_train, y_pred_train = torch.max(net(x_train),dim=1)
print(f"学習データの正解率:{accuracy_score(y_pred_train, y_train)}")
y_max_test, y_pred_test = torch.max(net(x_test),dim=1)
print(f"評価データの正解率:{accuracy_score(y_pred_test, y_test)}")
"""
学習データの正解率:0.5235866716585549
評価データの正解率:0.5396706586826348
"""