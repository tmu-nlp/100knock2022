#事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors 
from transformers import BertTokenizer, BertModel

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
max_len = 15
dw = 300 #埋め込みの次元
dh = 50 #隠れ層の次元
n_vocab = len(words) + 1 #語彙サイズ
PAD = 0 #padding_idx
epochs = 10

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate=0.04, otuput_size=4):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = torch.nn.Linear(768, otuput_size)  # BERTの出力に合わせて768次元を指定

    def forward(self, ids):
        seg_ids = torch.zeros_like(ids)
        attention_mask = (ids > 0)
        ret = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        last_hidden_state = ret["last_hidden_state"]
        x = last_hidden_state[:,0,:]
        logit = self.fc(x.view(-1,768))
        return logit

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
    pred = np.argmax(pred.to("cpu").data.numpy(), axis=1)
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
model = BERTClass()

#デバイスを指定
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model.to(device)

ds = TensorDataset(X_train.to(device), y_train.to(device)) #入力と正解ラベルをセットにする
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
epoch: 28
train loss: 1.7684916257858276, train acc: 0.43186072631973044
valid loss: 1.880683422088623, valid acc: 0.4124251497005988
epoch: 29
train loss: 1.6299771070480347, train acc: 0.4247472856608012
valid loss: 1.5980230569839478, valid acc: 0.43488023952095806
"""