#順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ．
#さらに，双方向RNNを多層化して実験せよ．
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

#双方向RNNにして多層化する
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True,bidirectional=True,num_layers=3)
        self.linear = torch.nn.Linear(dh*2,4)#カテゴリ数=4，双方向だから入力の次元2倍
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
train loss: 1.1249628067016602, train acc: 0.5190939722950206
valid loss: 1.1403045654296875, valid acc: 0.5059880239520959
epoch: 1
train loss: 1.0084437131881714, train acc: 0.6178397603893673
valid loss: 1.050950527191162, valid acc: 0.5838323353293413
epoch: 2
train loss: 0.8607656359672546, train acc: 0.698333957319356
valid loss: 0.9444324970245361, valid acc: 0.656437125748503
epoch: 3
train loss: 0.7676449418067932, train acc: 0.732029202545863
valid loss: 0.8773916363716125, valid acc: 0.6961077844311377
epoch: 4
train loss: 0.7416905164718628, train acc: 0.7358667165855485
valid loss: 0.8850452303886414, valid acc: 0.686377245508982
epoch: 5
train loss: 0.6716907024383545, train acc: 0.7511231748408835
valid loss: 0.8453795909881592, valid acc: 0.6931137724550899
epoch: 6
train loss: 0.6782166957855225, train acc: 0.7486896293523025
valid loss: 0.8741933703422546, valid acc: 0.6803892215568862
epoch: 7
train loss: 0.4961727261543274, train acc: 0.832366154998128
valid loss: 0.7356476187705994, valid acc: 0.7402694610778443
epoch: 8
train loss: 0.6161303520202637, train acc: 0.760389367278173
valid loss: 0.8866018056869507, valid acc: 0.6796407185628742
epoch: 9
train loss: 0.5305658578872681, train acc: 0.8037251965555972
valid loss: 0.8272125124931335, valid acc: 0.7230538922155688
"""