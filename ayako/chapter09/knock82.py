#確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題81で構築したモデルを学習せよ．
#訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．
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
ds = TensorDataset(X_train, y_train) #入力と正解ラベルをセットにする
loader = DataLoader(ds, batch_size=1, shuffle=True) #datasetを読み出し，バッチサイズで一回に読み出す数を指定できる
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
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        print("epoch: {}".format(epoch))
        print("train loss: {}, train acc: {}".format(loss.item(), accuracy(y_pred,y_train)))
        y_pred = model(X_valid)
        loss = loss_fn(y_pred, y_valid)
        print("valid loss: {}, valid acc: {}".format(loss.item(), accuracy(y_pred,y_valid)))

"""
epoch: 0
train loss: 2.906360387802124, train acc: 0.4112691875701984
valid loss: 3.0093002319335938, valid acc: 0.40793413173652693
epoch: 1
train loss: 3.298168659210205, train acc: 0.4055597154623736
valid loss: 3.4800503253936768, valid acc: 0.38323353293413176
epoch: 2
train loss: 2.8664746284484863, train acc: 0.3347997004867091
valid loss: 2.7768473625183105, valid acc: 0.3405688622754491
epoch: 3
train loss: 3.661273241043091, train acc: 0.4112691875701984
valid loss: 3.7412302494049072, valid acc: 0.4214071856287425
epoch: 4
train loss: 3.1218557357788086, train acc: 0.41922500935979035
valid loss: 3.1124486923217773, valid acc: 0.4311377245508982
epoch: 5
train loss: 2.500168800354004, train acc: 0.40546611755896667
valid loss: 2.41457462310791, valid acc: 0.4124251497005988
epoch: 6
train loss: 2.9045844078063965, train acc: 0.3577311868214152
valid loss: 3.0086939334869385, valid acc: 0.35179640718562877
epoch: 7
train loss: 3.069338798522949, train acc: 0.42718083114938227
valid loss: 3.132277727127075, valid acc: 0.4101796407185629
epoch: 8
train loss: 3.221982002258301, train acc: 0.3709284912017971
valid loss: 3.309067726135254, valid acc: 0.37949101796407186
epoch: 9
train loss: 2.746002435684204, train acc: 0.4167914638712093
valid loss: 2.8138957023620605, valid acc: 0.3967065868263473
"""