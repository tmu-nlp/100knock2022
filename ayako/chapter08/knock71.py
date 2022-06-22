import torch
import numpy as np

#学習データ読み込み
x_train = np.loadtxt("./data/x_train.txt", delimiter=" ")
#pytorchのtensor型にする(32bit浮動小数点)，サイズは10684×300
x_train = torch.tensor(x_train, dtype=torch.float32)
#print(x_train.size())

#重み行列Wはランダムな値で初期化
W = torch.rand(300, 4)

#softmaxで予測確率を得る，dim=1にすると行単位で合計1にしてくれる
softmax = torch.nn.Softmax(dim=1)
#matmulでtensorの行列積求める
print(softmax(torch.matmul(x_train[:1], W)))
print(softmax(torch.matmul(x_train[:4], W)))
"""
tensor([[0.1303, 0.2390, 0.2074, 0.4233]])
tensor([[0.1303, 0.2390, 0.2074, 0.4233],
        [0.2698, 0.1870, 0.3891, 0.1541],
        [0.2378, 0.1412, 0.5002, 0.1208],
        [0.2199, 0.1864, 0.3911, 0.2025]])
"""