#問題85や問題87のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，
#高性能なカテゴリ分類器を構築せよ．
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
epochs = 20

#双方向RNNにして多層化する
#epochs:10->20
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True,bidirectional=True,num_layers=5)
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
train loss: 1.096340298652649, train acc: 0.5452077873455634
valid loss: 1.1179453134536743, valid acc: 0.5239520958083832
epoch: 1
train loss: 0.9828696846961975, train acc: 0.6176525645825534
valid loss: 1.0404175519943237, valid acc: 0.5898203592814372
epoch: 2
train loss: 0.8747488856315613, train acc: 0.6819543242231374
valid loss: 0.9441144466400146, valid acc: 0.655688622754491
epoch: 3
train loss: 0.7540284395217896, train acc: 0.7351179333582928
valid loss: 0.8670346140861511, valid acc: 0.6946107784431138
epoch: 4
train loss: 0.6918440461158752, train acc: 0.7587046050168477
valid loss: 0.849901556968689, valid acc: 0.6976047904191617
epoch: 5
train loss: 0.7183506488800049, train acc: 0.7491576188693373
valid loss: 0.918057918548584, valid acc: 0.7028443113772455
epoch: 6
train loss: 0.6066983938217163, train acc: 0.7920254586297267
valid loss: 0.8298138976097107, valid acc: 0.7320359281437125
epoch: 7
train loss: 0.5120658278465271, train acc: 0.814582553350805
valid loss: 0.8074635863304138, valid acc: 0.7005988023952096
epoch: 8
train loss: 0.44192904233932495, train acc: 0.845282665668289
valid loss: 0.7608916759490967, valid acc: 0.7103293413173652
epoch: 9
train loss: 0.37436383962631226, train acc: 0.8683077499064021
valid loss: 0.7327985763549805, valid acc: 0.75
epoch: 10
train loss: 0.49316948652267456, train acc: 0.8214152002995133
valid loss: 0.8634241819381714, valid acc: 0.7267964071856288
epoch: 11
train loss: 0.2976147532463074, train acc: 0.8903968551104455
valid loss: 0.7415496706962585, valid acc: 0.7642215568862275
epoch: 12
train loss: 0.3009355068206787, train acc: 0.9016286035192812
valid loss: 0.7063966393470764, valid acc: 0.7544910179640718
epoch: 13
train loss: 0.3182010054588318, train acc: 0.883845001871958
valid loss: 0.837593674659729, valid acc: 0.7260479041916168
epoch: 14
train loss: 0.23872023820877075, train acc: 0.9165106701609884
valid loss: 0.8121084570884705, valid acc: 0.7754491017964071
epoch: 15
train loss: 0.19377626478672028, train acc: 0.9292399850243355
valid loss: 0.8581748604774475, valid acc: 0.7919161676646707
epoch: 16
train loss: 0.18234537541866302, train acc: 0.9401909397229502
valid loss: 0.8801963925361633, valid acc: 0.7327844311377245
epoch: 17
train loss: 0.11940708011388779, train acc: 0.9656495694496443
valid loss: 0.7869079113006592, valid acc: 0.7896706586826348
epoch: 18
train loss: 0.22310885787010193, train acc: 0.9219393485585923
valid loss: 1.05210542678833, valid acc: 0.7664670658682635
epoch: 19
train loss: 0.05496452376246452, train acc: 0.9865219019093973
valid loss: 0.9347822666168213, valid acc: 0.7866766467065869
"""