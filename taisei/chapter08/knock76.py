import torch
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X_train = joblib.load("./output/X_train.joblib")
Y_train = joblib.load("./output/Y_train.joblib")
X_valid = joblib.load("./output/X_valid.joblib")
Y_valid = joblib.load("./output/Y_valid.joblib")

X_train = np.array(list(X_train.values))
Y_train = Y_train.values
X_valid = np.array(list(X_valid.values))
Y_valid = Y_valid.values

X_train_tenso = torch.from_numpy(X_train.astype(np.float32))
Y_train_tenso = torch.from_numpy(Y_train.astype(np.int64))
X_valid_tenso = torch.from_numpy(X_valid.astype(np.float32))
Y_valid_tenso = torch.from_numpy(Y_valid.astype(np.int64))


torch.manual_seed(0)
net = torch.nn.Linear(300, 4)
loss_f = torch.nn.CrossEntropyLoss()

#最適化器：Optimizer
#訓練したいモデルのパラメータを登録。また、学習率をハイパラとして渡すことで初期化を行う
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


losses_train = []
losses_valid = []
accs_train = []
accs_valid = []
axis = 1

for i in tqdm(range(100)):
    Y_pred_train = net(X_train_tenso)
    loss_train = loss_f(Y_pred_train, Y_train_tenso)
    losses_train.append(loss_train.detach().numpy())

    Y_pred_valid = net(X_valid_tenso)
    loss_valid = loss_f(Y_pred_valid, Y_valid_tenso)
    losses_valid.append(loss_valid.detach().numpy())

    Y_max_train, Y_pred_train = torch.max(net(X_train_tenso), axis)
    acc_train = accuracy_score(Y_pred_train, Y_train_tenso)
    
    Y_max_valid, Y_pred_valid = torch.max(net(X_valid_tenso), axis)
    acc_valid = accuracy_score(Y_pred_valid, Y_valid_tenso)

    accs_train.append(acc_train)
    accs_valid.append(acc_valid)

    #バックプロぱゲーション
    optimizer.zero_grad() #パラメータの勾配をリセット
    loss_train.backward() #誤差逆伝播
    optimizer.step() #各パラメータの勾配を使用してパラメータの値を調整

    #学習途中のパラメータ
    torch.save(net.state_dict(), f"./output/knock76_checkpoints/model{i}.pth")
    #最適化アルゴリズムの内部状態
    torch.save(optimizer.state_dict(), f"./output/knock76_checkpoints/optime{i}.pth")

"""
fig1 = plt.figure()
plt.plot(losses_train, label="train loss")
plt.plot(losses_valid, label="valid loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("./output/knock76_loss.png")
fig2 = plt.figure()
plt.plot(accs_train, label="train acc")
plt.plot(accs_valid, label="valid acc")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("./output/knock76_accuracy.png")
"""

