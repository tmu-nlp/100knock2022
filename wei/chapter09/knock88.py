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
    LEARNING_RATES = [1e-3, 1e-2, 1e-1]
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    NUM_LAYERS = 2

    OUTPUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    BATCH_SIZEs = [32, 64, 128]
    NUM_EPOCHs = [10, 30]

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
    for NUM_EPOCH in NUM_EPOCHs:
        for BATCH_SIZE in BATCH_SIZEs:
            for LEARNING_RATE in LEARNING_RATES:
                # cnn
                criterion = torch.nn.CrossEntropyLoss()
                optimizer_cnn = torch.optim.SGD(cnn.parameters(), lr=LEARNING_RATE)
                train_model(dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCH, model=cnn,
                    optimizer=optimizer_cnn, criterion=criterion, device=device, collate_fn=Padsequence(PADDING_IDX))
                train_loss_cnn, train_acc_cnn = cal_loss_acc(cnn, criterion, dataset_train, device=device)
                valid_loss_cnn, valid_acc_cnn = cal_loss_acc(cnn, criterion, dataset_valid, device=device)
                print(f'performance of CNN:')
                print(f'train_loss:{train_loss_cnn:.4f}, train_acc:{train_acc_cnn:.4f}')
                print(f'valid_loss:{valid_loss_cnn:.4f}, valid_acc:{valid_acc_cnn:.4f}')

                # bi_rnn
                optimizer_birnn = torch.optim.SGD(birnn.parameters(), lr=LEARNING_RATE)
                train_model(dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCH, model=birnn,
                    optimizer=optimizer_birnn, criterion=criterion, device=device, collate_fn=Padsequence(PADDING_IDX))
                train_loss_rnn, train_acc_rnn = cal_loss_acc(birnn, criterion, dataset_train, device=device)
                valid_loss_rnn, valid_acc_rnn = cal_loss_acc(birnn, criterion, dataset_valid, device=device)
                print(f'performance of biRNN:')
                print(f'train_loss:{train_loss_rnn:.4f}, train_acc:{train_acc_rnn:.4f}')
                print(f'valid_loss:{valid_loss_rnn:.4f}, valid_acc:{valid_acc_rnn:.4f}')

'''
NUM_EPOCH:10  BATCH_SIZE:32  LEARNING_RATE:1e-3
performance of CNN:
train_loss:1.1026, train_acc:0.5675
valid_loss:1.1100, valid_acc:0.5525
performance of biRNN:
train_loss:1.1691, train_acc:0.4097
valid_loss:1.1713, valid_acc:0.4010

NUM_EPOCH:10  BATCH_SIZE:32  LEARNING_RATE:1e-2
performance of CNN:
train_loss:0.8098, train_acc:0.7249
valid_loss:0.8576, valid_acc:0.6927
performance of biRNN:
train_loss:1.1327, train_acc:0.5319
valid_loss:1.1481, valid_acc:0.5277

NUM_EPOCH:10  BATCH_SIZE:32  LEARNING_RATE:1e-1
performance of CNN:
train_loss:0.0851, train_acc:0.9860
valid_loss:0.4487, valid_acc:0.8501
performance of biRNN:
train_loss:1.0273, train_acc:0.6389
valid_loss:1.0392, valid_acc:0.6372

NUM_EPOCH:10  BATCH_SIZE:64  LEARNING_RATE:1e-3
performance of CNN:
train_loss:0.0745, train_acc:0.9909
valid_loss:0.4303, valid_acc:0.8516
performance of biRNN:
train_loss:0.9504, train_acc:0.6663
valid_loss:0.9663, valid_acc:0.6649

NUM_EPOCH:10  BATCH_SIZE:64  LEARNING_RATE:1e-2
performance of CNN:
train_loss:0.0573, train_acc:0.9952
valid_loss:0.4225, valid_acc:0.8561
performance of biRNN:
train_loss:0.9452, train_acc:0.6680
valid_loss:0.9765, valid_acc:0.6612
NUM_EPOCH:10  BATCH_SIZE:64  LEARNING_RATE:1e-1
performance of CNN:
train_loss:0.0278, train_acc:0.9983
valid_loss:0.4273, valid_acc:0.8553
performance of biRNN:
train_loss:1.0951, train_acc:0.5777
valid_loss:1.1035, valid_acc:0.5705

NUM_EPOCH:10  BATCH_SIZE:128  LEARNING_RATE:1e-3
performance of CNN:
train_loss:0.0271, train_acc:0.9983
valid_loss:0.4280, valid_acc:0.8561
performance of biRNN:
train_loss:1.0747, train_acc:0.5812
valid_loss:1.0820, valid_acc:0.5720

NUM_EPOCH:10  BATCH_SIZE:128  LEARNING_RATE:1e-2
performance of CNN:
train_loss:0.0246, train_acc:0.9985
valid_loss:0.4265, valid_acc:0.8546
performance of biRNN:
train_loss:1.0862, train_acc:0.5718
valid_loss:1.0982, valid_acc:0.5577

NUM_EPOCH:10  BATCH_SIZE:128  LEARNING_RATE:1e-1
performance of CNN:
train_loss:0.0183, train_acc:0.9987
valid_loss:0.4396, valid_acc:0.8583
performance of biRNN:
train_loss:1.1147, train_acc:0.5029
valid_loss:1.1075, valid_acc:0.5187

NUM_EPOCH:30  BATCH_SIZE:32  LEARNING_RATE:1e-3
performance of CNN:
train_loss:0.0173, train_acc:0.9988
valid_loss:0.4394, valid_acc:0.8561
performance of biRNN:
train_loss:1.0180, train_acc:0.6154
valid_loss:1.0217, valid_acc:0.6117

NUM_EPOCH:30  BATCH_SIZE:32  LEARNING_RATE:1e-2
performance of CNN:
train_loss:0.0133, train_acc:0.9991
valid_loss:0.4435, valid_acc:0.8561
performance of biRNN:
train_loss:0.8491, train_acc:0.7213
valid_loss:0.9206, valid_acc:0.6837

NUM_EPOCH:30  BATCH_SIZE:32  LEARNING_RATE:1e-1
performance of CNN:
train_loss:0.0103, train_acc:0.9980
valid_loss:0.5668, valid_acc:0.8568
performance of biRNN:
train_loss:1.2388, train_acc:0.4315
valid_loss:1.2539, valid_acc:0.4235

NUM_EPOCH:30  BATCH_SIZE:64  LEARNING_RATE:1e-3
performance of CNN:
train_loss:0.0063, train_acc:0.9993
valid_loss:0.5268, valid_acc:0.8538

NUM_EPOCH:30  BATCH_SIZE:64  LEARNING_RATE:1e-2
NUM_EPOCH:30  BATCH_SIZE:64  LEARNING_RATE:1e-1



NUM_EPOCH:30  BATCH_SIZE:64  LEARNING_RATE:1e-3
NUM_EPOCH:30  BATCH_SIZE:64  LEARNING_RATE:1e-2
NUM_EPOCH:30  BATCH_SIZE:64  LEARNING_RATE:1e-1

NUM_EPOCH:30  BATCH_SIZE:128  LEARNING_RATE:1e-3
NUM_EPOCH:30  BATCH_SIZE:128  LEARNING_RATE:1e-2
NUM_EPOCH:30  BATCH_SIZE:128  LEARNING_RATE:1e-1

'''