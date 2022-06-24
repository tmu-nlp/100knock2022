from sklearn import neural_network
from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, input_features, output_features):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_features, output_features)  # 層を定義
        nn.init.normal_(self.layer1.weight, 0.0, 1.0)  # 正規分布で初期化   https://blog.snowhork.com/2018/11/pytorch-initialize-weight
    
    def forward(self, input):
        """前に伝搬する関数？"""
        x = self.layer1(input)  # ただ層に通すだけ
        return x

if __name__ == "__main__":
    X_train = torch.load("tensor/X_train.pt")

    model = NeuralNetwork(300, 4)  # d = 300, n = 4

    y_hat_1 = torch.softmax(model(X_train[0]), dim=-1)
    Y_hat = torch.softmax(model(X_train[4:]), dim=-1)

    # print(y_hat_1)
    # print(Y_hat)


"""
tensor([0.1130, 0.3225, 0.3858, 0.1786], grad_fn=<SoftmaxBackward0>)
tensor([[8.3613e-01, 3.1508e-02, 8.2418e-02, 4.9943e-02],
        [1.5431e-01, 3.8786e-02, 7.7523e-01, 3.1668e-02],
        [3.3664e-01, 2.1241e-01, 3.2990e-01, 1.2105e-01],
        ...,
        [3.8570e-01, 2.8991e-01, 2.4020e-01, 8.4190e-02],
        [8.0245e-01, 2.6377e-02, 1.6468e-01, 6.4981e-03],
        [1.4836e-01, 3.6027e-01, 4.9082e-01, 5.4766e-04]],
       grad_fn=<SoftmaxBackward0>)
"""