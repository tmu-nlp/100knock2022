import torch
import numpy as np
from torch import nn
from tqdm.notebook import tqdm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def determine_seed(seed):
    '''乱数の固定用関数'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


determine_seed(42)


class dataset(torch.utils.data.Dataset):
    '''Datasetクラスの定義, len, getitemが必須
    データローダを使う場合や前処理定義する場合'''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label


class SimpleNet(nn.Module):
    '''単層ニューラルネットワークモデルの定義
    class torch.nn.ModuleはすべてのNNモジュールの基底クラス、これのサブクラスによりモデルを定義'''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size,
                            bias=False)  # Linear...全結合層(重みパラメータの作成)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)  # 正規乱数で重みを初期化

    def forward(self, x):
        '''順方向の計算処理'''
        x = self.fc(x)
        return x


# 訓練データの読み込み
X_train = torch.load('./X_train.pt')
y_train = torch.load('./y_train.pt')
train_dataset = dataset(X_train, y_train)

# モデル構築
model = SimpleNet(300, 4)

# 事例 x1
# クロスエントロピー誤差の計算(推論と正解の確率分布がどれくい離れているか, 似ている程小さい値)
criterion = nn.CrossEntropyLoss()
l_1 = criterion(model(X_train[:1]), y_train[:1])

# 勾配の計算
#model.zero_grad()  # zero_grad(): 勾配を0で初期化
l_1.backward()  # 勾配を計算

print(f'損失: {l_1:.4f}')
print(f'勾配:\n{model.fc.weight.grad}')


# 事例集合 {x1, x2, x3, x4}
loss = criterion(model(X_train[:4]), y_train[:4])
# model.zero_grad()
loss.backward()

print(f'損失: {loss:.4f}')
print(f'勾配:\n{model.fc.weight.grad}')

'''
損失: 0.0870
勾配:
tensor([[-7.4401e-03,  2.0941e-03,  5.6887e-03,  ...,  4.6586e-03,
          3.1886e-04,  7.6620e-03],
        [ 1.9661e-03, -5.5338e-04, -1.5033e-03,  ..., -1.2311e-03,
         -8.4263e-05, -2.0248e-03],
        [ 4.7821e-03, -1.3460e-03, -3.6564e-03,  ..., -2.9943e-03,
         -2.0495e-04, -4.9247e-03],
        [ 6.9189e-04, -1.9474e-04, -5.2901e-04,  ..., -4.3322e-04,
         -2.9653e-05, -7.1253e-04]])
損失: 1.8612
勾配:
tensor([[ 0.0201,  0.0078,  0.0103,  ...,  0.0372,  0.0374,  0.0321],
        [ 0.0058,  0.0057, -0.0069,  ...,  0.0047,  0.0213, -0.0006],
        [-0.0168, -0.0058,  0.0020,  ..., -0.0096, -0.0124, -0.0277],
        [-0.0091, -0.0077, -0.0054,  ..., -0.0323, -0.0463, -0.0039]])
'''
