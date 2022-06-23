import torch
from torch import nn

X_train = torch.load('./X_train.pt')
y_train = torch.load('./y_train.pt')
#print(X_train.shape)
#print(y_train.shape)

W = torch.randn(300, 4)
softmax = torch.nn.Softmax(dim=1)
y_hat1 = softmax(torch.matmul(X_train[:1], W))  # torch.matmulで特徴ベクトルと重みの行列積を計算
Y_hat = softmax(torch.matmul(X_train[:4], W))
print(y_hat1)
print(Y_hat)
