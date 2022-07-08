'''
83. ミニバッチ化・GPU上での学習
batch_size事例ごとに損失・勾配を計算して学習を行える
'''

from knock80 import *    # made ids for words
from knock81 import *    # defined the RNN model and Dataset
from knock82 import cal_loss_acc, train_model    # defined how to train the model
import torch
import time


# very large to train

class Padsequence():
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sorted_batch = sorted(
            batch, key=lambda x : x['inputs'].shape[0],reverse=True
        )
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(
            sequences=sequences, batch_first=True, padding_value=self.padding_idx
        )
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])

        return {'inputs': sequences_padded, 'labels': labels}




if __name__ == '__main__':
    # set parameters
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    # define gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RNN(emb_size=EMB_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=OUTPUT_SIZE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    train_log = train_model(
        dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, model=model,
        optimizer=optimizer, criterion=criterion, device=device, collate_fn=Padsequence(PADDING_IDX)
    )


    train_loss, train_acc = cal_loss_acc(model, criterion, dataset_train)
    valid_loss, valid_acc = cal_loss_acc(model, criterion, dataset_valid)
    print(f'train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}')
    print(f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}')




