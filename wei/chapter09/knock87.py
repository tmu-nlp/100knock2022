""'''
87. 確率的勾配降下法によるCNNの学習
SGDを用いて、訓練データ上の損失と正解率、評価データ上の損失と正解率を表示しながら
問題86で構築したCNNを学習
'''
from knock82 import cal_loss_acc, train_model
from knock83 import Padsequence
from knock86 import *
import torch

if __name__ == '__main__':
    # set parameter
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUTPUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHTS = torch.load('knock84_weights.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    cnn = CNN(
        emb_size=EMB_SIZE, vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=OUTPUT_SIZE,
        output_chanels=OUTPUT_CHANNELS, kernel_heights=KERNEL_HEIGHTS, stride=STRIDE, padding=PADDING,
        emb_weights=WEIGHTS
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LEARNING_RATE)
    train_model(dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, model=cnn,
                optimizer=optimizer, criterion=criterion, device=device, collate_fn=Padsequence(PADDING_IDX))
    train_loss, train_acc = cal_loss_acc(cnn, criterion, dataset_train, device=device)
    valid_loss, valid_acc = cal_loss_acc(cnn, criterion, dataset_valid, device=device)
    print(f'train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}')
    print(f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}')


    '''
epoch:1, loss_train:1.3101, acc_train:0.4099, loss_valid:1.3051, acc_valid:0.4108, time_used:2.4333
epoch:2, loss_train:1.2504, acc_train:0.4736, loss_valid:1.2435, acc_valid:0.4738, time_used:0.6554
epoch:3, loss_train:1.2165, acc_train:0.4916, loss_valid:1.2097, acc_valid:0.4978, time_used:0.6772
epoch:4, loss_train:1.1967, acc_train:0.5004, loss_valid:1.1908, acc_valid:0.4940, time_used:0.9473
epoch:5, loss_train:1.1842, acc_train:0.5052, loss_valid:1.1795, acc_valid:0.4963, time_used:0.6791
epoch:6, loss_train:1.1750, acc_train:0.5115, loss_valid:1.1715, acc_valid:0.5030, time_used:0.6951
epoch:7, loss_train:1.1675, acc_train:0.5183, loss_valid:1.1651, acc_valid:0.5120, time_used:0.7602
epoch:8, loss_train:1.1610, acc_train:0.5219, loss_valid:1.1597, acc_valid:0.5187, time_used:0.6864
epoch:9, loss_train:1.1550, acc_train:0.5266, loss_valid:1.1546, acc_valid:0.5180, time_used:0.7156
epoch:10, loss_train:1.1495, acc_train:0.5291, loss_valid:1.1500, acc_valid:0.5202, time_used:0.7073
train_loss:1.1495, train_acc:0.5291
valid_loss:1.1500, valid_acc:0.5202
'''