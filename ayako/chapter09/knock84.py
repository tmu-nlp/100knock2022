#事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）
#での学習済み単語ベクトル）で単語埋め込みemb(x)を初期化し，学習せよ．
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors 

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
PAD = len(words) #padding_idx
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

#事前学習済みの単語ベクトルでembを初期化
w2v = KeyedVectors.load_word2vec_format("../chapter07/GoogleNews-vectors-negative300.bin.gz", binary=True)#学習済みモデル読み込み
for key, val in d.items():#単語辞書{key:word，val:id}
    if key in w2v.vocab:#学習済みモデル辞書にその単語があるとき
        model.emb.weight[val].data = torch.tensor(w2v[key], dtype=torch.float32)#その単語の重みを学習済みモデルの値で定義
model.emb.weight = torch.nn.Parameter(model.emb.weight)#学習済みの重みに更新

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
train loss: 1.1352733373641968, train acc: 0.5079558217895919
valid loss: 1.1626821756362915, valid acc: 0.4663173652694611
epoch: 1
train loss: 1.0827172994613647, train acc: 0.5530700112317484
valid loss: 1.1228524446487427, valid acc: 0.5202095808383234
epoch: 2
train loss: 1.00046706199646, train acc: 0.613253463122426
valid loss: 1.0654983520507812, valid acc: 0.5726047904191617
epoch: 3
train loss: 0.9416483640670776, train acc: 0.652002995132909
valid loss: 1.0244367122650146, valid acc: 0.6062874251497006
epoch: 4
train loss: 0.9386447668075562, train acc: 0.6561213028828154
valid loss: 1.0575793981552124, valid acc: 0.6152694610778443
epoch: 5
train loss: 0.8094342350959778, train acc: 0.7194870834893299
valid loss: 0.9316867589950562, valid acc: 0.6766467065868264
epoch: 6
train loss: 0.7856426239013672, train acc: 0.7235117933358293
valid loss: 0.9359948635101318, valid acc: 0.6721556886227545
epoch: 7
train loss: 0.6968458294868469, train acc: 0.7640396855110445
valid loss: 0.8727753162384033, valid acc: 0.7050898203592815
epoch: 8
train loss: 0.6690267324447632, train acc: 0.7709659303631599
valid loss: 0.8692534565925598, valid acc: 0.7050898203592815
epoch: 9
train loss: 0.7303510904312134, train acc: 0.7305316360913515
valid loss: 0.9408360123634338, valid acc: 0.6669161676646707
"""