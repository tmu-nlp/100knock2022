""'''
88. パラメータチューニング
問題85: biCNN や問題87: CNN のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，
高性能なカテゴリ分類器を構築せよ．
'''
from knock80 import *    # made ids for words
from knock81 import *    # defined the RNN model and Dataset
from knock82 import cal_loss_acc, train_model
from knock83 import Padsequence
import torch
from knock85 import biRNN
from knock86 import CNN


if __name__ == '__main__':
    # set parameter
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATES = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    NUM_LAYERS = 2

    OUTPUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    WEIGHTS = torch.load('knock84_weights.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    birnn = biRNN(
        hidden_size=HIDDEN_SIZE, emb_size=EMB_SIZE, vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX,
    output_size=OUTPUT_SIZE, num_layers=NUM_LAYERS, device=device, emb_weight=WEIGHTS, bidirectional=True
    )

    cnn = CNN(
        emb_size=EMB_SIZE, vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=OUTPUT_SIZE,
        output_chanels=OUTPUT_CHANNELS, kernel_heights=KERNEL_HEIGHTS, stride=STRIDE, padding=PADDING,
        emb_weights=WEIGHTS
    )
    for LEARNING_RATE in LEARNING_RATES:
        # cnn
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_cnn = torch.optim.SGD(cnn.parameters(), lr=LEARNING_RATE)
        train_model(dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, model=cnn,
                    optimizer=optimizer_cnn, criterion=criterion, device=device, collate_fn=Padsequence(PADDING_IDX))
        train_loss_cnn, train_acc_cnn = cal_loss_acc(cnn, criterion, dataset_train, device=device)
        valid_loss_cnn, valid_acc_cnn = cal_loss_acc(cnn, criterion, dataset_valid, device=device)
        print(f'train_loss:{train_loss_cnn:.4f}, train_acc:{train_acc_cnn:.4f}')
        print(f'valid_loss:{valid_loss_cnn:.4f}, valid_acc:{valid_acc_cnn:.4f}')

        # bi_rnn
        optimizer_birnn = torch.optim.SGD(birnn.parameters(), lr=LEARNING_RATE)
        train_model(dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, model=birnn,
                    optimizer=optimizer_birnn, criterion=criterion, device=device, collate_fn=Padsequence(PADDING_IDX))
        train_loss_rnn, train_acc_rnn = cal_loss_acc(birnn, criterion, dataset_train, device=device)
        valid_loss_rnn, valid_acc_rnn = cal_loss_acc(birnn, criterion, dataset_valid, device=device)
        print(f'train_loss:{train_loss_rnn:.4f}, train_acc:{train_acc_rnn:.4f}')
        print(f'valid_loss:{valid_loss_rnn:.4f}, valid_acc:{valid_acc_rnn:.4f}')

