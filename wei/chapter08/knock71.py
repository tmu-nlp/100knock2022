'''
71. 単層ニューラルネットワークによる予測
重み行列W: (d×L),randomly initialized
'''

import torch
from torch import nn



# 単層ニューラルネットワークを定義
class sglNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)  # (tensor, mean, std),正規乱数で重みを初期化

    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # load all tensors onto CPU. also use map_location=lambda storage, loc:storage.cuda(1) to load tentors onto GPU 1
    X_train = torch.load('X_train.pt', map_location=torch.device('cpu'))
    print(X_train.size())     # torch.Size([10672, 300])

    my_nn = sglNN(300, 4)
    # 事例x_1を分類した時、各カテゴリに属する確率を表すベクトル
    y_hat_1 = torch.softmax(my_nn(X_train[:1]), dim=-1)
    print(y_hat_1)        # tensor([[0.0244, 0.1171, 0.3714, 0.4871]], grad_fn=<SoftmaxBackward>)

    Y_hat = torch.softmax(my_nn.forward(X_train[:4]), dim=-1)
    print(Y_hat)
    '''
    tensor([[0.0244, 0.1171, 0.3714, 0.4871],
        [0.1205, 0.0938, 0.6176, 0.1681],
        [0.0433, 0.0390, 0.3749, 0.5429],
        [0.1530, 0.2091, 0.2542, 0.3838]], grad_fn=<SoftmaxBackward>)
    '''



