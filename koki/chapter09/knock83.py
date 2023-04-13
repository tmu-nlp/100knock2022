import torch
from torch import nn
from knock80 import *
from knock81 import *
from knock82 import *


class Padsequence():
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        '''__call__ インスタンス化したオブジェクトを関数のように呼び出すことで呼び出されるメソッド'''
        item = {}
        sorted_batch = sorted(
            batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])
        item['inputs'] = sequences_padded
        item['labels'] = labels
        return item


# パラメータの設定
VOCAB_SIZE = len(set(word_id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word_id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 5e-2

# モデル定義
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

# 損失関数定義
criterion = nn.CrossEntropyLoss()

# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# cuda
device = torch.device('cuda:0')

# 学習(cuda)
# log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion,
# optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

# 学習(cpu)
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion,
                  optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX))

# 損失ログの可視化
plt.plot(np.array(log['train']).T[0], label='train')
plt.plot(np.array(log['valid']).T[0], label='valid')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.savefig('./results/output83_loss.png')
plt.show()

# 正解率ログの可視化
plt.plot(np.array(log['train']).T[1], label='train')
plt.plot(np.array(log['valid']).T[1], label='valid')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
# plt.savefig('./results/output83_accuracy.png')
plt.show()

# 正解率の算出
_, acc_train = calculate_loss_and_accuracy(model, dataset_train)
_, acc_test = calculate_loss_and_accuracy(model, dataset_test)
print(f'正解率（学習データ）: {acc_train:.3f}')
print(f'正解率（評価データ）: {acc_test:.3f}')

'''
正解率（学習データ）: 0.611
正解率（評価データ）: 0.573
'''
