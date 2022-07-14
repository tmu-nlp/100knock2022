from gensim.models import KeyedVectors

# 学習済みモデルのロード
model = KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin.gz', binary=True)
print(model['United_States'])


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
        # emb.size() = (batch_size, seq_len, emb_size)
        out, hidden = self.rnn(emb, hidden)
        # out.size() = (batch_size, seq_len, hidden_size * num_directions)
        out = self.fc(out[:, -1, :])
        # out.size() = (batch_size, output_size)
        return out

    def init_hidden(self):
        hidden = torch.zeros(
            self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        return hidden


# パラメータの設定
VOCAB_SIZE = len(set(word_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
NUM_LAYERS = 1
LEARNING_RATE = 5e-2
BATCH_SIZE = 32
NUM_EPOCHS = 10

# モデルの定義
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE,
            HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# デバイスの指定
device = torch.device('cuda')

# モデルの学習
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion,
                  optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

# 損失ログの可視化
plt.plot(np.array(log['train']).T[0], label='train')
plt.plot(np.array(log['valid']).T[0], label='valid')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.savefig('./results/output84_loss.png')
plt.show()

# 正解率ログの可視化
plt.plot(np.array(log['train']).T[1], label='train')
plt.plot(np.array(log['valid']).T[1], label='valid')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
# plt.savefig('./results/output84_accuracy.png')
plt.show()

# 正解率の算出
_, acc_train = calculate_loss_and_accuracy(model, dataset_train)
_, acc_test = calculate_loss_and_accuracy(model, dataset_test)
print(f'正解率（学習データ）: {acc_train:.3f}')
print(f'正解率（評価データ）: {acc_test:.3f}')

'''
正解率（学習データ）: 0.640
正解率（評価データ）: 0.621
'''
