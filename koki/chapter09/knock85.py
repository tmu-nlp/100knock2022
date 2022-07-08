import torch
from torch import nn
from knock80 import word_id
from knock81 import dataset_train, dataset_valid
from knock82 import train_model
from knock83 import Padsequence


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size,
                 hidden_size, num_layers, emb_weights=None, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1  # 単方向：1、双方向：2
        if emb_weights != None:  # 指定があれば埋め込み層の重みをemb_weightsで初期化
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(
                vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers,
                          nonlinearity='tanh', bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden()  # h0のゼロベクトルを作成
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self):
        hidden = torch.zeros(
            self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        return hidden


class Bidirectional_RNN(RNN):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size,
                 hidden_size, num_layers, emb_weights=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # 双方向なので2
        if emb_weights != None:  # 指定があれば埋め込み層の重みをemb_weightsで初期化
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(
                vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers,
                          nonlinearity='tanh', bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    super().forward()
    super().init_hidden()


# パラメータ
VOCAB_SIZE = len(set(word_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
NUM_LAYERS = 2  # 双方向なので層の数も2層に設定
BATCH_SIZE = 32
NUM_EPOCHS = 10


# モデル定義
model = Bidirectional_RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE,
                          HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights)

# 損失関数定義(クロスエントロピー)
criterion = nn.CrossEntropyLoss()

# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)

# cuda
device = torch.device('cuda')

# 学習
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion,
                  optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

# 損失ログの可視化
plt.plot(np.array(log['train']).T[0], label='train')
plt.plot(np.array(log['valid']).T[0], label='valid')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.savefig('./results/output85_loss.png')
plt.show()

# 正解率ログの可視化
plt.plot(np.array(log['train']).T[1], label='train')
plt.plot(np.array(log['valid']).T[1], label='valid')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
# plt.savefig('./results/output85_accuracy.png')
plt.show()

# 正解率の算出
_, acc_train = calculate_loss_and_accuracy(model, dataset_train)
_, acc_test = calculate_loss_and_accuracy(model, dataset_test)
print(f'正解率（学習データ）: {acc_train:.3f}')
print(f'正解率（評価データ）: {acc_test:.3f}')

'''
正解率（学習データ）: 0.640
正解率（評価データ）: 0.640
'''
