#問題82のコードを改変し，B事例ごとに損失・勾配を計算して学習を行えるようにせよ（Bの値は適当に選べ）．
#また，GPU上で学習を実行せよ．
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader

#データを読み込む
train = pd.read_csv('../chapter06/train.txt', header=None, sep='\t')
valid = pd.read_csv('../chapter06/valid.txt', header=None, sep='\t')
test = pd.read_csv('../chapter06/test.txt', header=None, sep='\t') 

#単語をidに変換して辞書作成
#ref:https://stmind.hatenablog.com/entry/2020/07/04/173131
vectorizer = CountVectorizer(min_df=2) #CountVectorizerのインスタンス生成
train_title = train.iloc[:,0].str.lower() #小文字化してタイトルのみを抽出
cnt = vectorizer.fit_transform(train_title).toarray() #各単語出現頻度を各文ごとにカウントして2次元配列を返す
sm = cnt.sum(axis=0) #sumで単語ごとのカウント(1次元配列)を得る
idx = np.argsort(sm)[::-1] #単語カウント配列を頻度順にソートして降順で取得
words = np.array(vectorizer.get_feature_names())[idx] #頻度順の単語リスト

d = dict() #単語id辞書
for i in range(len(words)):
  d[words[i]] = i+1 #d[word] = id(1スタート)

#各文の単語をidに変換してidリストを返す
def get_id(sentence):
    r = []
    for word in sentence:
        r.append(d.get(word,0))#単語辞書からそのwordのidを取得．ない場合は0を返す
    return r

#Dataframeから各文ごとにidリストになっているリストを返す
def df2id(df):
    ids = []
    for i in df.iloc[:,0].str.lower():
        ids.append(get_id(i.split()))
    return ids

#ハイパラ設定
max_len = 10
dw = 300 #埋め込みの次元
dh = 50 #隠れ層の次元
n_vocab = len(words) + 1 #語彙サイズ
PAD = len(words) #？？？語彙サイズより1少ないやつ
epochs = 10

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True)
        self.linear = torch.nn.Linear(dh,4)#カテゴリ数=4
        self.softmax = torch.nn.Softmax()

    def forward(self, x, h=None):
        x = self.emb(x)#入力単語id列xの埋め込み
        y, h = self.rnn(x, h)#予測ラベルyと次の隠れ状態
        y = self.linear(y[:,-1,:])
        return y

#idリストをtensorに変換
def list2tensor(data, max_len):
    new = []
    for d in data:
        if len(d) > max_len:#max文長以上の時はmax_lenの系列長にする
            d = d[:max_len]
        else:#max文長以下の時はpaddingで埋めて系列長を揃える
            d += [PAD] * (max_len - len(d))
        new.append(d)
    return torch.tensor(new, dtype=torch.int64)

#正解率を計算
def accuracy(pred, label):
    pred = np.argmax(pred.data.numpy(), axis=1)
    label = label.data.numpy()
    return (pred == label).mean()

#Dataframeの各要素をidリストに変換
X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

#idリストをtensorに変換
X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

#chapter08で作成したラベルを読み込んでtensorに変換
y_train = np.loadtxt('../chapter08/data/y_train.txt')
y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = np.loadtxt('../chapter08/data/y_valid.txt')
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = np.loadtxt('../chapter08/data/y_test.txt')
y_test = torch.tensor(y_test, dtype=torch.int64)

#モデルを定義
model = RNN()
#デバイスを指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device)) #入力と正解ラベルをセットにする

#バッチサイズを適当に選ぶ．大きい方が効率は良いぽい.
loader = DataLoader(ds, batch_size=128, shuffle=True) #datasetを読み出し，バッチサイズで一回に読み出す数を指定できる
loss_fn = torch.nn.CrossEntropyLoss() #損失関数は交差エントロピー
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1) #確率的勾配降下法を使用

#モデルを学習
#ref:https://ohke.hateblo.jp/entry/2019/12/07/230000
for epoch in range(epochs):
    for xx, yy in loader:
        y_pred = model(xx) #予測
        loss = loss_fn(y_pred, yy) #損失(誤差)を計算
        optimizer.zero_grad() #勾配をゼロクリアして初期化
        loss.backward() #誤差を逆伝播(このメソッドは勾配が蓄積される)
        optimizer.step() #パラメータを更新
    with torch.no_grad():
        y_pred = model(X_train.to(device))
        loss = loss_fn(y_pred, y_train.to(device))
        print("epoch: {}".format(epoch))
        print("train loss: {}, train acc: {}".format(loss.item(), accuracy(y_pred,y_train)))
        y_pred = model(X_valid.to(device))
        loss = loss_fn(y_pred, y_valid.to(device))
        print("valid loss: {}, valid acc: {}".format(loss.item(), accuracy(y_pred,y_valid)))
"""
epoch: 0
train loss: 1.1439059972763062, train acc: 0.5001871958068139
valid loss: 1.1651636362075806, valid acc: 0.47979041916167664
epoch: 1
train loss: 1.1011861562728882, train acc: 0.5343504305503557
valid loss: 1.1383172273635864, valid acc: 0.4940119760479042
epoch: 2
train loss: 1.0360050201416016, train acc: 0.582459752901535
valid loss: 1.097566843032837, valid acc: 0.531437125748503
epoch: 3
train loss: 0.951221764087677, train acc: 0.6430175964058406
valid loss: 1.047037959098816, valid acc: 0.5980538922155688
epoch: 4
train loss: 0.8757261037826538, train acc: 0.690565331336578
valid loss: 0.9914066791534424, valid acc: 0.6452095808383234
epoch: 5
train loss: 0.889188289642334, train acc: 0.6719393485585923
valid loss: 1.0156047344207764, valid acc: 0.625748502994012
epoch: 6
train loss: 0.7742186188697815, train acc: 0.7321228004492699
valid loss: 0.909437358379364, valid acc: 0.6908682634730539
epoch: 7
train loss: 0.8145917654037476, train acc: 0.7052602021714713
valid loss: 0.970932126045227, valid acc: 0.6459580838323353
epoch: 8
train loss: 0.6797890663146973, train acc: 0.7620741295394983
valid loss: 0.8631077408790588, valid acc: 0.6923652694610778
epoch: 9
train loss: 0.6486903429031372, train acc: 0.7743354548858106
valid loss: 0.8528627753257751, valid acc: 0.7103293413173652
"""