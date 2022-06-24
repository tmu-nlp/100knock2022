'''
72.　損失と勾配を計算
事例に対して、クロスエントロピー損失と、行列Wに対する勾配を計算
'''
from knock71 import *
from torch import nn

if __name__ == '__main__':
    X_train = torch.load('X_train.pt')
    print(X_train.size())       # torch.Size([10672, 300])
    y_train = torch.load('y_train.pt')
    print(y_train.size())          # torch.Size([10672])

    my_nn = sglNN(300, 4)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # 入力はSoftMax前の値で、1事例に対して、損失を計算
    l_1 = criterion(my_nn(X_train[:1]), y_train[:1])
    # 勾配をゼロで初期化
    my_nn.zero_grad()
    # 逆伝播で勾配を計算
    l_1.backward()
    print(f'loss: {l_1:.4f}\ngradient:\n{my_nn.fc.weight.grad}')

    l = criterion(my_nn(X_train[:4]), y_train[:4])
    my_nn.zero_grad()
    l.backward()
    print(f'loss: {l:.4f}\ngradient:\n{my_nn.fc.weight.grad}')

'''
loss: 0.8988
gradient:
tensor([[-0.0200, -0.0586, -0.0030,  ...,  0.0135,  0.0103,  0.0043],
        [ 0.0012,  0.0035,  0.0002,  ..., -0.0008, -0.0006, -0.0003],
        [ 0.0014,  0.0042,  0.0002,  ..., -0.0010, -0.0007, -0.0003],
        [ 0.0174,  0.0509,  0.0026,  ..., -0.0117, -0.0090, -0.0038]])
loss: 1.8650
gradient:
tensor([[ 3.5865e-03, -1.5381e-02,  2.0558e-03,  ..., -8.7670e-03,
          3.2369e-04, -7.5328e-05],
        [ 6.7728e-04,  3.3191e-03,  1.2339e-03,  ..., -3.8808e-03,
          1.8752e-03,  1.2036e-04],
        [ 1.5574e-03, -2.2226e-02, -1.0416e-02,  ...,  4.2238e-02,
         -2.2675e-02, -1.0866e-03],
        [-5.8212e-03,  3.4288e-02,  7.1265e-03,  ..., -2.9590e-02,
          2.0476e-02,  1.0416e-03]])
'''
