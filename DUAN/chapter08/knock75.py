import string
import torch
import numpy as np
import pandas as pd
from torch import nn
from matplotlib import pyplot as plt
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# データの読み込み、抽出、分割
df = pd.read_csv('./100knock2022/DUAN/chapter06/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]
train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])
# ファイルのロード
model = KeyedVectors.load_word2vec_format('./100knock2022/DUAN/chapter08/GoogleNews-vectors-negative300.bin', binary=True)

def transform_w2v(text):
    # 記号をスペースに置換する
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    # スペースで分割してリスト化
    words = text.translate(table).split()
    # ベクトル化  
    vec = [model[word] for word in words if word in model]  
    # 平均ベクトルをTensor型に変換する
    return torch.tensor(sum(vec) / len(vec))

# 特徴ベクトル
X_train = torch.stack([transform_w2v(text) for text in train['TITLE']])
X_valid = torch.stack([transform_w2v(text) for text in valid['TITLE']])
X_test = torch.stack([transform_w2v(text) for text in test['TITLE']])
# ラベルベクトル
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

class SLPNet(nn.Module): # 単層ニューラルネットワークを定義する
    def __init__(self, input_size, output_size): # ネットワークを構成するレイヤーを定義する
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0) # 正規乱数で重みを初期化する

    def forward(self, x): # インプットデータが順伝播時に通るレイヤーを順に配置する
        x = self.fc(x)
        return x

class NewsDataset(Dataset):
    def __init__(self, X, y): # datasetの構成要素を指定する
        self.X = X
        self.y = y
    # 返す値を指定する
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx):  
        return [self.X[idx], self.y[idx]]

# DatasetとDataloaderの作成
dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

# 損失と正解率を計算する
def calculate_loss_and_accuracy(model, criterion, loader):
    model.eval() # 評価モードに設定する
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            # 順伝播
            outputs = model(inputs)
            # 損失計算
            loss += criterion(outputs, labels).item()
            # 正解率計算
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss / len(loader), correct / total

# 単層ニューラルネットワークの初期化
model = SLPNet(300, 4)
# 損失関数を定義する
criterion = nn.CrossEntropyLoss()
# オプティマイザを定義する
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
# 学習
log_train = []
log_valid = []
epochs = 10
for epoch in range(epochs):
    # 訓練モードに設定する
    model.train()
    for inputs, labels in dataloader_train:
        # 勾配をゼロで初期化する
        optimizer.zero_grad()
        # 順伝播、誤差逆伝播、重み更新
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # 損失と正解率の算出
    loss_train, acc_train = calculate_loss_and_accuracy(model, criterion, dataloader_train)
    loss_valid, acc_valid = calculate_loss_and_accuracy(model, criterion, dataloader_valid)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])
    # ログを出力する
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}')  

# グラフへのプロット
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(np.array(log_train).T[0], label='train')
ax[0].plot(np.array(log_valid).T[0], label='valid')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend() # 凡例を表示する
ax[1].plot(np.array(log_train).T[1], label='train')
ax[1].plot(np.array(log_valid).T[1], label='valid')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend()
plt.show()

'''
epoch: 1, loss_train: 0.3254, accuracy_train: 0.8852, loss_valid: 0.3527, accuracy_valid: 0.8793
epoch: 2, loss_train: 0.2892, accuracy_train: 0.9006, loss_valid: 0.3175, accuracy_valid: 0.8943
epoch: 3, loss_train: 0.2734, accuracy_train: 0.9081, loss_valid: 0.3041, accuracy_valid: 0.8981
epoch: 4, loss_train: 0.2590, accuracy_train: 0.9114, loss_valid: 0.2912, accuracy_valid: 0.8996
epoch: 5, loss_train: 0.2504, accuracy_train: 0.9149, loss_valid: 0.2824, accuracy_valid: 0.9040
epoch: 6, loss_train: 0.2451, accuracy_train: 0.9162, loss_valid: 0.2792, accuracy_valid: 0.9033
epoch: 7, loss_train: 0.2404, accuracy_train: 0.9172, loss_valid: 0.2780, accuracy_valid: 0.9018
epoch: 8, loss_train: 0.2374, accuracy_train: 0.9183, loss_valid: 0.2758, accuracy_valid: 0.9040
epoch: 9, loss_train: 0.2379, accuracy_train: 0.9181, loss_valid: 0.2791, accuracy_valid: 0.8996
epoch: 10, loss_train: 0.2300, accuracy_train: 0.9220, loss_valid: 0.2687, accuracy_valid: 0.9078
'''