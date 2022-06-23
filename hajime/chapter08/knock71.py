from torch import nn
import torch


class SingleLayerPerceptronNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.fc(x)
        return x


X_train = torch.load("X_train.pt")
model = SingleLayerPerceptronNetwork(300, 4)
y_hat_1 = torch.softmax(model(X_train[:1]), dim=-1)
Y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)

# print(y_hat_1)
# print(Y_hat)

"""
tensor([[0.0202, 0.0639, 0.1269, 0.7889]], grad_fn=<SoftmaxBackward0>)
tensor([[0.0202, 0.0639, 0.1269, 0.7889],
        [0.1149, 0.2677, 0.1351, 0.4824],
        [0.0073, 0.0120, 0.8173, 0.1634],
        [0.0213, 0.0369, 0.1437, 0.7982]], grad_fn=<SoftmaxBackward0>)
"""
