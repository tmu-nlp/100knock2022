from tqdm import tqdm
import torch, time
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)#シード固定しとく

#学習/評価データ読み込み
x_train = np.loadtxt("./data/x_train.txt", delimiter=" ")
y_train = np.loadtxt("./data/y_train.txt")
x_valid = np.loadtxt("./data/x_valid.txt", delimiter=" ")
y_valid = np.loadtxt("./data/y_valid.txt")
#pytorchのtensor型にする
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
x_valid = torch.tensor(x_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.int64)

#ネットワーク作成
net = torch.nn.Linear(300, 4, bias=False)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
batchsize = [2**i for i in range(5)]

#データをタイトルとラベルの組ごとにタプルとして扱う
dataset = TensorDataset(x_train, y_train)

times = []

for bs in batchsize:
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)#ミニバッチを返してくれる
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in tqdm(range(10)):
        start = time.time()#時間計測スタート
        for xx, yy in loader:
            optimizer.zero_grad()#勾配を0で初期化
            y_pred = net(x_train)#確率を計算
            loss = loss_fn(y_pred, y_train)#損失を計算
            loss.backward()#勾配を計算
            optimizer.step()#最適化ステップを実行

        #損失の記録
        train_losses.append(loss.detach().numpy())
        valid_losses.append(loss_fn(net(x_valid), y_valid).detach().numpy())
        #カテゴリの予測
        y_max_train, y_pred_train = torch.max(net(x_train),dim=1)
        y_max_valid, y_pred_valid = torch.max(net(x_valid),dim=1)
        #正解率の記録
        train_acc = accuracy_score(y_pred_train, y_train)
        valid_acc = accuracy_score(y_pred_valid, y_valid)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        #学習時間計測
        times.append(time.time() - start)

        #モデルを保存
        #torch.save(net.state_dict(), f"checkpoints/model_epoch{epoch}.pt")

#計測時間を出力
with open ("output/time.txt", "w") as f:
    for i in range(len(times)):
        if i%10 == 0:
            print(f"---------bachsize={2**(i/10)}----------", file=f)
        print(f"epoch{i%10+1}:{times[i]}", file=f)

"""
#損失の変化をプロット
fig = plt.figure()
plt.plot(train_losses, label="train loss")
plt.plot(valid_losses, label="valid loss")
plt.legend()#凡例
plt.savefig("output/loss.png")

#正解率の変化をプロット
fig = plt.figure()
plt.plot(train_accs, label="train acc")
plt.plot(valid_accs, label="valid acc")
plt.legend()#凡例
plt.savefig("output/acc.png")
"""