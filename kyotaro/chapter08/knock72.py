import torch
from torch import nn
from knock71 import NeuralNetwork

loss = nn.CrossEntropyLoss()  # クロスエントロピーを定義

X_train = torch.load("tensor/X_train.pt")
y_train = torch.load("tensor/Y_train.pt")

model = NeuralNetwork(X_train.shape[1], 4)  # knock71では300次元であったが、今回はX_trainに合わせる

l_1 = loss(model(X_train[0]), y_train[0])  # 損失関数の計算
model.zero_grad()  # 勾配の初期化
l_1.backward()  # 勾配の計算（誤差逆伝播？）

print(f'損失 : {l_1:.4f}')
print(f'勾配 :\n{model.layer1.weight.grad}')  # 層の重みの勾配を出力


l = loss(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()

print(f'損失 : {l:.4f}')
print(f'勾配 : {model.layer1.weight.grad}')

"""
損失 : 1.5233
勾配 :
tensor([[ 0.0080, -0.0058, -0.0052,  ..., -0.0134, -0.0098,  0.0097],
        [ 0.0313, -0.0230, -0.0204,  ..., -0.0528, -0.0385,  0.0380],
        [-0.0584,  0.0428,  0.0380,  ...,  0.0984,  0.0717, -0.0709],
        [ 0.0191, -0.0140, -0.0124,  ..., -0.0321, -0.0234,  0.0231]])
"""