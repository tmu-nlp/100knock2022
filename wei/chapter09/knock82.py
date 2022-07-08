""'''
82. 確率的勾配降下法による学習
'''
import numpy as np
from knock80 import *    # made ids for words
from knock81 import *    # defined the RNN model and Dataset
import torch
from torch.utils.data import DataLoader
import time
from matplotlib import pyplot as plt


def cal_loss_acc(model, criterion, dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    loss = float(0)
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss/len(dataloader), correct/total


def train_model(dataset_train, dataset_valid, batch_size, num_epochs, model, optimizer, criterion, device, collate_fn):
    model.to(device)

    d_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # epoch毎にshuffled
    d_loader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False, drop_last=False)

    log_train = []
    log_valid = []

    for epoch in range(1, num_epochs+1):
        start = time.time()

        model.train()
        for data in d_loader_train:
            optimizer.zero_grad()

            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end = time.time()

        model.eval()
        loss_train, acc_train = cal_loss_acc(model, criterion, dataset_train, device=device)
        loss_valid, acc_valid = cal_loss_acc(model, criterion, dataset_valid, device=device)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # save checkpoints
        # model_param_dic = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dic': optimizer.state_dict()}
        # torch.save(model_param_dic, f'knock82_checkpoint_{epoch}.pth')

        time_used = end - start

        print(f'epoch:{epoch}, loss_train:{loss_train:.4f}, acc_train:{acc_train:.4f}, loss_valid:{loss_valid:.4f}, acc_valid:{acc_valid:.4f}, time_used:{time_used:.4f}')

    return {'train': log_train, 'valid': log_valid}

# visualizaztion
def visualize_logs(log):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log['train']).T[0], label='train')
    ax[0].plot(np.array(log['valid']).T[0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log['train']).T[1], label='train')
    ax[1].plot(np.array(log['valid']).T[1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.savefig("82.png")

if __name__ == '__main__':
    # set parameter
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1
    NUM_EPOCHS = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RNN(emb_size=EMB_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE, padding_idx=PADDING_IDX, output_size=OUTPUT_SIZE,device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    train_log = train_model(dataset_train, dataset_valid, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, model=model,optimizer=optimizer, criterion=criterion, device=device, collate_fn=None)
    visualize_logs(train_log)

    train_loss, train_acc = cal_loss_acc(model, criterion, dataset_train)
    valid_loss, valid_acc = cal_loss_acc(model, criterion, dataset_valid)
    print(f'train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}')
    print(f'valid_loss:{valid_loss:.4f}, valid_acc:{valid_acc:.4f}')

    '''
epoch:1, loss_train:1.0913, acc_train:0.5436, loss_valid:1.1119, acc_valid:0.5112, time_used:115.9502
epoch:2, loss_train:0.9804, acc_train:0.6152, loss_valid:1.0253, acc_valid:0.5750, time_used:117.7669
epoch:3, loss_train:0.7855, acc_train:0.7174, loss_valid:0.8545, acc_valid:0.6919, time_used:128.2161
epoch:4, loss_train:0.6359, acc_train:0.7705, loss_valid:0.7273, acc_valid:0.7481, time_used:138.1814
epoch:5, loss_train:0.5341, acc_train:0.8040, loss_valid:0.6525, acc_valid:0.7646, time_used:138.7537
epoch:6, loss_train:0.4658, acc_train:0.8251, loss_valid:0.6292, acc_valid:0.7759, time_used:132.5480
epoch:7, loss_train:0.4090, acc_train:0.8475, loss_valid:0.6122, acc_valid:0.7736, time_used:146.0526
epoch:8, loss_train:0.3550, acc_train:0.8693, loss_valid:0.5744, acc_valid:0.7939, time_used:148.0779
epoch:9, loss_train:0.3268, acc_train:0.8851, loss_valid:0.5532, acc_valid:0.8066, time_used:139.2923
epoch:10, loss_train:0.3015, acc_train:0.8906, loss_valid:0.5906, acc_valid:0.7916, time_used:137.9502
train_loss:0.3015, train_acc:0.8906
valid_loss:0.5906, valid_acc:0.7916
'''


