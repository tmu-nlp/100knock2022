#ID番号で表現された単語列x=(x1,x2,…,xT)がある.
#ただし,Tは単語列の長さ,xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）.
#再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い,
#単語列xからカテゴリyを予測するモデルとして,次式を実装せよ.

#実装だけでOK?
from torch import nn

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, padding_idx=0, hidden_dim=50, category_size=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx)#padding_idxは各データの単語数を揃えるために空きを埋めるindex
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        #batch_firstをTrueにすると，(seq_len, batch, input_size)と指定されている入力テンソルの型を(batch, seq_len, input_size)にできる
        self.linear = nn.Linear(hidden_dim, category_size)
        self.softmax = nn.Softmax(bias=True)
    
    def forward(self, x, h=None):
        emb = self.emb(x)#入力単語id列xの埋め込み
        y, h_out = self.rnn(emb, h)#予測ラベルyと次の隠れ状態
        y = y[:,-1,:]
        y = self.linear(y)
        y = self.softmax(y)
        return y
