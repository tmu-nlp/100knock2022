# ×
import torch
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

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

"""
#GPUかCPUを指定。モデルおよび入力TensorをGPUに送るために
device1 = torch.device("cuda")
device2 = torch.device("cpu")
"""

torch.manual_seed(0)
net = torch.nn.Linear(300, 4)
loss_f = torch.nn.CrossEntropyLoss()

#最適化器：Optimizer
#訓練したいモデルのパラメータを登録。また、学習率をハイパラとして渡すことで初期化を行う
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

axis = 1

#エポックごとにシャッフルするからデータと正解をまとめる
train_data = torch.utils.data.TensorDataset(X_train_tenso, Y_train_tenso)

for batch in [2 ** i for i in range(5)]:
    losses_train = []
    losses_valid = []
    accs_train = []
    accs_valid = []
    times = []
    for i in tqdm(range(100)):
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
        start = time.time()
        sum_loss_train = 0
        for batch_x, batch_y in train_data:
            Y_pred_train = net(batch_x)
            loss_train = loss_f(Y_pred_train, batch_y)
            optimizer.zero_grad() #パラメータの勾配をリセット
            loss_train.backward() #誤差逆伝播
            optimizer.step() #各パラメータの勾配を使用してパラメータの値を調整
            sum_loss_train += loss_train.item()

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



        end = time.time()
        times.append(end - start)
        """
        #学習途中のパラメータ
        torch.save(net.state_dict(), f"./output/knock77_checkpoints/batch{batch}/model{i}.pth")
        #最適化アルゴリズムの内部状態
        torch.save(optimizer.state_dict(), f"./output/knock77_checkpoints/batch{batch}/optime{i}.pth")
        """
    print(f'----------batch={batch}----------')
    print(f'1エポックにかかる時間の平均：{sum(times)/len(times):.5f}')
    print(f'訓練データの損失：{losses_train[-1]:.5f}')
    print(f'開発データの損失：{losses_valid[-1]:.5f}')
    print(f'訓練データの正解率：{accs_train[-1]:.5f}')
    print(f'開発データの正解率：{accs_valid[-1]:.5f}')

"""
fig1 = plt.figure()
plt.plot(losses_train, label="train loss")
plt.plot(losses_valid, label="valid loss")
plt.legend()
plt.savefig("./output/knock77_loss.png")
fig2 = plt.figure()
plt.plot(accs_train, label="train acc")
plt.plot(accs_valid, label="valid acc")
plt.legend()
plt.savefig("./output/knock77_accuracy.png")
"""

