import torch
import numpy as np

#学習データ読み込み
x_train = np.loadtxt("./data/x_train.txt", delimiter=" ")
y_train = np.loadtxt("./data/y_train.txt")
#pytorchのtensor型にする
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)

#ネットワークを作成
#Linear(入力サイズ，出力サイズ)の線形変換
#重み行列Wとして扱いたいからバイアスはなしにする
net = torch.nn.Linear(300, 4, bias=False) 

#forwardで予測する
y_pred1 = torch.softmax(net.forward(x_train[:1]), dim=1)
y_pred4 = torch.softmax(net.forward(x_train[:4]), dim=1)

#クロスエントロピー損失loss(input, target)
loss = torch.nn.CrossEntropyLoss()
loss1 = loss(y_pred1, y_train[:1])
loss4 = loss(y_pred4, y_train[:4])

#backwardで勾配計算
#事例1の方
net.zero_grad()#勾配を0で初期化
loss1.backward()
print(f"事例1のクロスエントロピー損失: {loss1}")
print(f"事例1の勾配: {net.weight.grad}")
#事例1~4の方
net.zero_grad()#勾配を0で初期化
loss4.backward()
print(f"事例1~4のクロスエントロピー損失: {loss4}")
print(f"事例1~4の勾配: {net.weight.grad}")

"""
事例1のクロスエントロピー損失: 1.372583031654358
事例1の勾配: tensor([[-0.0494, -0.0201,  0.0041,  ..., -0.0113, -0.0091,  0.0427],
        [ 0.0152,  0.0062, -0.0013,  ...,  0.0035,  0.0028, -0.0132],
        [ 0.0164,  0.0067, -0.0014,  ...,  0.0038,  0.0030, -0.0142],
        [ 0.0177,  0.0072, -0.0015,  ...,  0.0040,  0.0033, -0.0153]])
事例1~4のクロスエントロピー損失: 1.3808194398880005
事例1~4の勾配: tensor([[-0.0108, -0.0023, -0.0091,  ..., -0.0055, -0.0012,  0.0193],
        [ 0.0090,  0.0018, -0.0013,  ...,  0.0007,  0.0034, -0.0027],
        [ 0.0037,  0.0040,  0.0102,  ...,  0.0012, -0.0002, -0.0081],
        [-0.0019, -0.0035,  0.0003,  ...,  0.0036, -0.0020, -0.0085]])
"""