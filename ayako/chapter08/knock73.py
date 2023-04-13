from tqdm import tqdm
import torch
import numpy as np

#学習データ読み込み
x_train = np.loadtxt("./data/x_train.txt", delimiter=" ")
y_train = np.loadtxt("./data/y_train.txt")
#pytorchのtensor型にする
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)

#ネットワーク作成
net = torch.nn.Linear(300, 4, bias=False)
loss_fn = torch.nn.CrossEntropyLoss()

#確率的勾配降下法SGD(モデルのパラメータ，学習率)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

#100epochで学習終了
losses = []
for epoch in tqdm(range(100)):
    optimizer.zero_grad()#勾配を0で初期化
    y_pred = torch.softmax(net.forward(x_train), dim=1)
    loss = loss_fn(y_pred, y_train)#損失を計算
    loss.backward()#勾配を計算
    optimizer.step()#最適化ステップを実行
    losses.append(loss)

#学習したモデルを保存
torch.save(net.state_dict(), "model.pt")