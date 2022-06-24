from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
        return [feature, label]


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


# データの読み込み
X_train, y_train = torch.load('./X_train.pt'), torch.load('./y_train.pt')
X_valid, y_valid = torch.load('./X_valid.pt'), torch.load('./y_valid.pt')
X_test, y_test = torch.load('./X_test.pt'), torch.load('./y_test.pt')

# Datasetの作成
train_dataset = dataset(X_train, y_train)
valid_dataset = dataset(X_valid, y_valid)
test_dataset = dataset(X_test, y_test)

# Dataloaderの作成 (Datasetからサンプルを取得して、ミニバッチを作成する)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=len(valid_dataset), shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=True)

model = SimpleNet(300, 4)
criterion = nn.CrossEntropyLoss()

# 最適化手法の選択、SGD(Stochastic Gradient Decent : 確率的勾配降下法 ... 求めた勾配方向にその大きさだけパラメータを更新する)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

num_epochs = 100  # エポック上限
for epoch in range(num_epochs):
    model.train()  # 訓練モード, Moduleは training/evaluation の2種類のモードを持っている。
    loss_train = 0.

    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()  # 損失を記録
    
    loss_train = loss_train / i  # バッチごとの平均損失

    model.eval()  # 評価モード, dropoutやbatch norm の切り替え
    with torch.no_grad():
        inputs, labels = next(iter(valid_dataloader))
        outputs = model(inputs)
        loss_valid = criterion(outputs, labels)

    if epoch % 10 == 0:
        print(f'epoch: {epoch} \t loss_train: {loss_train:.4f} \t loss_valid: {loss_valid:.4f}')

'''
epoch: 0 	 loss_train: 0.4693 	 loss_valid: 0.3609
epoch: 10 	 loss_train: 0.2257 	 loss_valid: 0.3207
epoch: 20 	 loss_train: 0.2151 	 loss_valid: 0.3186
epoch: 30 	 loss_train: 0.2099 	 loss_valid: 0.3263
epoch: 40 	 loss_train: 0.2084 	 loss_valid: 0.3321
epoch: 50 	 loss_train: 0.2068 	 loss_valid: 0.3465
epoch: 60 	 loss_train: 0.2057 	 loss_valid: 0.3390
epoch: 70 	 loss_train: 0.2048 	 loss_valid: 0.3394
epoch: 80 	 loss_train: 0.2045 	 loss_valid: 0.3448
epoch: 90 	 loss_train: 0.2043 	 loss_valid: 0.3533
'''
