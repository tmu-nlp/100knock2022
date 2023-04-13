import torch
import numpy as np
import joblib
from tqdm import tqdm
#0-6 最適化

X_train = joblib.load("./output/X_train.joblib")
Y_train = joblib.load("./output/Y_train.joblib")

X_train = np.array(list(X_train.values))
Y_train = Y_train.values

X_train_tenso = torch.from_numpy(X_train.astype(np.float32))
Y_train_tenso = torch.from_numpy(Y_train.astype(np.int64))

torch.manual_seed(0)
net = torch.nn.Linear(300, 4)
loss_f = torch.nn.CrossEntropyLoss()

#最適化器：Optimizer
#訓練したいモデルのパラメータを登録。また、学習率をハイパラとして渡すことで初期化を行う
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for _ in tqdm(range(1000)):
    # Y_pred = torch.softmax(net.forward(X_train_tenso), dim=-1)
    Y_pred = net(X_train_tenso)
    loss_X = loss_f(Y_pred, Y_train_tenso)

    #バックプロぱゲーション
    optimizer.zero_grad() #パラメータの勾配をリセット
    loss_X.backward() #誤差逆伝播
    optimizer.step() #各パラメータの勾配を使用してパラメータの値を調整

#pytorchのモデルは学習したパラメータを内部に状態辞書（state_dict）として保存する
#torch.saveにより永続化できる
torch.save(net.state_dict(), "./output/knock73_model.pth")
