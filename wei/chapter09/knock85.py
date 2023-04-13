'''
85. 双方向RNN・多層化
順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習
h_t: 時刻ｔの隠れ状態ベクトル
RNN(x,h)は入力xと次時刻の隠れ状態hから前状態を計算するRNNユニット
W^(yh)∈R^{L×2dh}は隠れ状態ベクトルからカテゴリを予測するための行列
b^{(y)}∈R^{L}はバイアス項である
biRNNを実験
https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
'''
from knock80 import *    # made ids for words
from knock81 import *    # defined the RNN model and Dataset, also make ids for word, myDataset and labels' tensor for 4 categories
from knock82 import cal_loss_acc, train_model
from knock83 import Padsequence
import torch
from torch import nn

# define biRNN model
class biRNN(nn.Module):
    def __init__(self, hidden_size, emb_size, vocab_size, padding_idx, output_size, num_layers, device, emb_weight=None, bidirectional=False):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size    # output_dim
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1               # to be bidirectional
        if emb_weight != None:
            self.emb = nn.Embedding.from_pretrained(
                emb_weight, padding_idx=padding_idx
            )
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(
            emb_size, hidden_size, num_layers=num_layers,
            nonlinearity='tanh', batch_first=True, bidirectional=bidirectional
        )
        # batch_first: if True, then input and output tensors has shape of (batch_size, seq_len, feature_dim) and does not apply to hidden/cell.
        #この設定により、複数のRNN_cellをひとまとめにする。
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        #全結合層で,(input_dim, output_dim)を入れて、tensor of shape (num_inputs, output_dim) を返す

    # forward learning
    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = torch.zeros(
            self.num_layers*self.num_directions,
            self.batch_size, self.hidden_size, device=self.device)
        x = self.emb(x)
        y, hidden = self.rnn(x, hidden)
        # x: 入力で、tensor of shape (batch_size, seq_len, input_dim)
        # in_hidden: 隠れ層の初期値で、tensor of shape (directs * num_layers, batch_size, hidden_size)
        # y: 次の層への出力で、tensor of shape (batch_size, seq_len, directs*hidden_size)
        # out_hidden: 現在の隠れ層の状態で、tensor of shape (directs*num_layers, batch_size, hidden_size)
        y = self.fc(y[:, -1, :])
        # (batch_size, seq_len, output_dim)を－1で指定し、最後の時刻の出力だけを入れて、(num_inputs, output_dim) を返す
        return y

if __name__ == '__main__':
    # set parameter
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    NUM_LAYERS = 2    # Stacking 2 RNNs together to form a stacked RNN, with 2nd RNN taking in outputs of 1st RNN
    WEIGHTS = torch.load('knock84_weights.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bi_rnn = biRNN(
        hidden_size=HIDDEN_SIZE, emb_size=EMB_SIZE, vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX,
    output_size=OUTPUT_SIZE, num_layers=NUM_LAYERS, device=device, emb_weight=WEIGHTS, bidirectional=True
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(bi_rnn.parameters(), lr=LEARNING_RATE)
    train_log = train_model(dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, model=bi_rnn,
                            optimizer=optimizer, criterion=criterion, device=device, collate_fn=Padsequence(PADDING_IDX))

    train_loss, train_acc = cal_loss_acc(bi_rnn, criterion, dataset_train, device=device)
    valid_loss, valid_acc = cal_loss_acc(bi_rnn, criterion, dataset_valid, device=device)
    print(f'train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}')
    print(f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}')



    '''
epoch:1, loss_train:1.2544, acc_train:0.4229, loss_valid:1.2504, acc_valid:0.4228, time_used:2.6662
epoch:2, loss_train:1.1949, acc_train:0.4273, loss_valid:1.1925, acc_valid:0.4310, time_used:2.0254
epoch:3, loss_train:1.1761, acc_train:0.4249, loss_valid:1.1742, acc_valid:0.4318, time_used:2.0022
epoch:4, loss_train:1.1703, acc_train:0.4251, loss_valid:1.1686, acc_valid:0.4318, time_used:2.0430
epoch:5, loss_train:1.1682, acc_train:0.4253, loss_valid:1.1666, acc_valid:0.4333, time_used:2.0351
epoch:6, loss_train:1.1673, acc_train:0.4254, loss_valid:1.1657, acc_valid:0.4355, time_used:2.0231
epoch:7, loss_train:1.1668, acc_train:0.4258, loss_valid:1.1652, acc_valid:0.4303, time_used:1.9742
epoch:8, loss_train:1.1664, acc_train:0.4256, loss_valid:1.1649, acc_valid:0.4370, time_used:1.9947
epoch:9, loss_train:1.1665, acc_train:0.4284, loss_valid:1.1650, acc_valid:0.4318, time_used:2.0055
epoch:10, loss_train:1.1668, acc_train:0.4291, loss_valid:1.1653, acc_valid:0.4258, time_used:2.0517
train_loss:1.1668, train_acc:0.4291
valid_loss:1.1653, valid_acc:0.4258
'''