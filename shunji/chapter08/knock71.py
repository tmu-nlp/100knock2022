import torch
from torch import nn
import pickle


class SLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=1.0)  # 正規乱数で重みを初期化

    def forward(self, x):
        """順伝播メソッド"""
        x = self.fc(x)
        return x


X_train = torch.load("X_train.pt")  # 行列の読み込み
model = SLP(300, 4)  # 単層パーセプトロンの初期化
with open('SLP.pkl', 'wb') as f:
    pickle.dump(model, f)


y_hat_1 = torch.softmax(model(X_train[0]), dim=0)
print(y_hat_1)

Y_hat = torch.softmax(model.forward(X_train[:4]), dim=1)
print(Y_hat)
