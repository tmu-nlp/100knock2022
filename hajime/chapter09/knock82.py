from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
from torch import optim
from knock81 import *
import numpy as np
from matplotlib import pyplot as plt


def calc_loss_acc(model, dataset, device=None, criterion=None):
    # model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            outputs = model(inputs)

            if criterion != None:
                loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

        return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion,  optimizer, num_epochs, device=None):
    model.to(device)

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=1, shuffle=False)
    # dataloader_valid = DataLoader(
    #     dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    log_train = []
    log_valid = []

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        loss_train = 0.0
        for data in dataloader_train:
            optimizer.zero_grad()

            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        model.eval()
        loss_train, acc_train = calc_loss_acc(
            model, dataset_train, device, criterion=criterion)
        loss_valid, acc_valid = calc_loss_acc(
            model, dataset_valid, device, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(
        ), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, train_time: {(end_time - start_time):.4f}sec')
    return {
        "train": log_train,
        "valid": log_valid
    }


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


if __name__ == "__main__":
    VOCAB_SIZE = len(set(w2id.id_dict.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(w2id.id_dict.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1
    NUM_EPOCHS = 10

    model = RNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    log = train_model(dataset_train, dataset_valid, BATCH_SIZE,
                      model, criterion, optimizer, NUM_EPOCHS)
    visualize_logs(log)
    _, acc_train = calc_loss_acc(model, dataset_train)
    _, acc_test = calc_loss_acc(model, dataset_test)
    print(f'accuracy (train)：{acc_train:.3f}')
    print(f'accuracy (test)：{acc_test:.3f}')

"""
epoch: 1, loss_train: 1.0794, accuracy_train: 0.5554, loss_valid: 1.0992, accuracy_valid: 0.5479, train_time: 25.4308sec
epoch: 2, loss_train: 1.0188, accuracy_train: 0.5926, loss_valid: 1.0624, accuracy_valid: 0.5734, train_time: 26.0320sec
epoch: 3, loss_train: 0.9452, accuracy_train: 0.6395, loss_valid: 1.0025, accuracy_valid: 0.6085, train_time: 27.3790sec
epoch: 4, loss_train: 0.8186, accuracy_train: 0.7040, loss_valid: 0.9068, accuracy_valid: 0.6751, train_time: 25.2013sec
epoch: 5, loss_train: 0.7032, accuracy_train: 0.7484, loss_valid: 0.8082, accuracy_valid: 0.7186, train_time: 24.4388sec
epoch: 6, loss_train: 0.6357, accuracy_train: 0.7713, loss_valid: 0.7657, accuracy_valid: 0.7320, train_time: 24.5022sec
epoch: 7, loss_train: 0.5496, accuracy_train: 0.7997, loss_valid: 0.7237, accuracy_valid: 0.7537, train_time: 26.0339sec
epoch: 8, loss_train: 0.5320, accuracy_train: 0.8063, loss_valid: 0.7714, accuracy_valid: 0.7373, train_time: 25.4809sec
epoch: 9, loss_train: 0.4833, accuracy_train: 0.8229, loss_valid: 0.7738, accuracy_valid: 0.7478, train_time: 25.4647sec
epoch: 10, loss_train: 0.5506, accuracy_train: 0.7988, loss_valid: 0.9134, accuracy_valid: 0.7103, train_time: 26.6588sec

accuracy (train)：0.799
accuracy (test)：0.703
"""
