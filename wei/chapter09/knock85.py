'''
85. 双方向RNN・多層化
順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習
h_t: 時刻ｔの隠れ状態ベクトル
RNN(x,h)は入力xと次時刻の隠れ状態hから前状態を計算するRNNユニット
W^(yh)∈R^{L×2dh}は隠れ状態ベクトルからカテゴリを予測するための行列
b^{(y)}∈R^{L}はバイアス項である
biRNNを実験
'''
from knock80 import *    # made ids for words
from knock81 import *    # defined the RNN model and Dataset
from knock82 import cal_loss_acc, train_model
from knock83 import Padsequence
import torch
from torch import nn

# define biRNN model
class biRNN(nn.Module):
    def __init__(self, hidden_size, emb_size, vocab_size, padding_idx, output_size, num_layers, device, emb_weight=None, bidirectional=False):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1  # to be bidiretional
        if emb_weight != None:
            self.emb = nn.Embedding.from_pretrained(
                emb_weight, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(
            emb_size, hidden_size, num_layers=num_layers,
            nonlinearity='tanh', batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    # forward learning
    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = torch.zeros(
            self.num_layers*self.num_directions,
            self.batch_size, self.hidden_size, device=self.device)
        x = self.emb(x)
        y, hidden = self.rnn(x, hidden)
        y = self.fc(y[:, -1, :])
        return y

if __name__ == '__main__':
    # set parameter
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    NUM_LAYERS = 2
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
    epoch:1, loss_train:1.3216, acc_train:0.4170, loss_valid:1.3146, acc_valid:0.4130, time_used:2.6366
epoch:2, loss_train:1.2738, acc_train:0.4269, loss_valid:1.2677, acc_valid:0.4273, time_used:1.9337
epoch:3, loss_train:1.2405, acc_train:0.4296, loss_valid:1.2350, acc_valid:0.4325, time_used:1.8560
epoch:4, loss_train:1.2172, acc_train:0.4308, loss_valid:1.2121, acc_valid:0.4303, time_used:1.8448
epoch:5, loss_train:1.2012, acc_train:0.4297, loss_valid:1.1963, acc_valid:0.4288, time_used:1.8478
epoch:6, loss_train:1.1903, acc_train:0.4268, loss_valid:1.1855, acc_valid:0.4303, time_used:2.1103
epoch:7, loss_train:1.1831, acc_train:0.4289, loss_valid:1.1784, acc_valid:0.4370, time_used:1.9324
epoch:8, loss_train:1.1784, acc_train:0.4278, loss_valid:1.1736, acc_valid:0.4430, time_used:1.9514
epoch:9, loss_train:1.1754, acc_train:0.4261, loss_valid:1.1706, acc_valid:0.4370, time_used:2.0002
epoch:10, loss_train:1.1732, acc_train:0.4265, loss_valid:1.1685, acc_valid:0.4415, time_used:1.8507
train_loss:1.1732, train_acc:0.4265
valid_loss:1.1685, valid_acc:0.4415
'''